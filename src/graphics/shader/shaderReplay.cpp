#include "graphics/shader/shaderReplay.h"

#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <type_traits>

namespace Libs::Graphics::ShaderReplay {
namespace {

constexpr std::array<char, 8> CaptureMagic          = {'K', 'Y', 'T', 'Y', 'C', 'S', 'R', 'P'};
constexpr std::array<char, 8> PixelCaptureMagic     = {'K', 'Y', 'T', 'Y', 'P', 'S', 'R', 'P'};
constexpr uint32_t            CaptureVersion        = 1;
constexpr uint32_t            MaxCaptureWords       = 16u * 1024u * 1024u;
constexpr uint32_t            MaxCaptureDescriptors = 1u * 1024u * 1024u;

bool Fail(std::string* error, const std::string& message) {
	if (error != nullptr) {
		*error = message;
	}
	return false;
}

class Writer {
public:
	explicit Writer(const std::filesystem::path& file)
	    : m_out(file, std::ios::binary | std::ios::trunc) {}

	[[nodiscard]] bool Valid() const { return static_cast<bool>(m_out); }

	bool Bytes(const void* data, size_t size) {
		m_out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
		return static_cast<bool>(m_out);
	}

	bool U32(uint32_t value) {
		const std::array<uint8_t, 4> bytes = {
		    static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
		    static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u)};
		return Bytes(bytes.data(), bytes.size());
	}

	bool U64(uint64_t value) {
		return U32(static_cast<uint32_t>(value)) && U32(static_cast<uint32_t>(value >> 32u));
	}

private:
	std::ofstream m_out;
};

class Reader {
public:
	explicit Reader(const std::filesystem::path& file): m_in(file, std::ios::binary) {}

	[[nodiscard]] bool Valid() const { return static_cast<bool>(m_in); }

	bool Bytes(void* data, size_t size) {
		m_in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
		return static_cast<bool>(m_in);
	}

	bool U32(uint32_t& value) {
		std::array<uint8_t, 4> bytes {};
		if (!Bytes(bytes.data(), bytes.size())) {
			return false;
		}
		value = static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8u |
		        static_cast<uint32_t>(bytes[2]) << 16u | static_cast<uint32_t>(bytes[3]) << 24u;
		return true;
	}

	bool U64(uint64_t& value) {
		uint32_t lo = 0;
		uint32_t hi = 0;
		if (!U32(lo) || !U32(hi)) {
			return false;
		}
		value = static_cast<uint64_t>(lo) | static_cast<uint64_t>(hi) << 32u;
		return true;
	}

	[[nodiscard]] bool AtEnd() { return m_in.peek() == std::ifstream::traits_type::eof(); }

private:
	std::ifstream m_in;
};

bool WriteWords(Writer& writer, std::span<const uint32_t> words) {
	if (words.size() > std::numeric_limits<uint32_t>::max() ||
	    !writer.U32(static_cast<uint32_t>(words.size()))) {
		return false;
	}
	for (const auto word: words) {
		if (!writer.U32(word)) {
			return false;
		}
	}
	return true;
}

bool ReadWords(Reader& reader, std::vector<uint32_t>& words, uint32_t limit) {
	uint32_t count = 0;
	if (!reader.U32(count) || count > limit) {
		return false;
	}
	words.resize(count);
	for (auto& word: words) {
		if (!reader.U32(word)) {
			return false;
		}
	}
	return true;
}

bool WriteDescriptors(Writer&                                                writer,
                      std::span<const ShaderRecompiler::IR::DescriptorValue> values) {
	if (values.size() > std::numeric_limits<uint32_t>::max() ||
	    !writer.U32(static_cast<uint32_t>(values.size()))) {
		return false;
	}
	for (const auto& value: values) {
		if (value.dword_count > value.dwords.size() || !writer.U32(value.dword_count)) {
			return false;
		}
		for (const auto word: value.dwords) {
			if (!writer.U32(word)) {
				return false;
			}
		}
	}
	return true;
}

bool ReadDescriptors(Reader& reader, std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
	uint32_t count = 0;
	if (!reader.U32(count) || count > MaxCaptureDescriptors) {
		return false;
	}
	values.resize(count);
	for (auto& value: values) {
		if (!reader.U32(value.dword_count) || value.dword_count > value.dwords.size()) {
			return false;
		}
		for (auto& word: value.dwords) {
			if (!reader.U32(word)) {
				return false;
			}
		}
	}
	return true;
}

bool WriteInput(Writer& writer, const ShaderComputeInputInfo& input) {
	for (const auto value: input.threads_num) {
		if (!writer.U32(value)) return false;
	}
	for (const auto value: input.dispatch_threads_num) {
		if (!writer.U32(value)) return false;
	}
	if (!writer.U32(input.lds_size_dwords) || !writer.U32(input.scratch_size_dwords)) return false;
	for (const auto value: input.group_id) {
		if (!writer.U32(value ? 1u : 0u)) return false;
	}
	return writer.U32(input.dispatch_thread_dimensions ? 1u : 0u) &&
	       writer.U32(input.needs_lds_barriers ? 1u : 0u) && writer.U32(input.wave_size) &&
	       writer.U32(static_cast<uint32_t>(input.thread_ids_num)) &&
	       writer.U32(static_cast<uint32_t>(input.workgroup_register)) &&
	       writer.U32(input.tg_size_en ? 1u : 0u);
}

bool ReadFlag(Reader& reader, bool& value) {
	uint32_t raw = 0;
	if (!reader.U32(raw) || raw > 1u) return false;
	value = raw != 0u;
	return true;
}

bool ReadInput(Reader& reader, ShaderComputeInputInfo& input) {
	for (auto& value: input.threads_num) {
		if (!reader.U32(value)) return false;
	}
	for (auto& value: input.dispatch_threads_num) {
		if (!reader.U32(value)) return false;
	}
	if (!reader.U32(input.lds_size_dwords) || !reader.U32(input.scratch_size_dwords)) return false;
	for (auto& value: input.group_id) {
		if (!ReadFlag(reader, value)) return false;
	}
	uint32_t thread_ids_num     = 0;
	uint32_t workgroup_register = 0;
	if (!ReadFlag(reader, input.dispatch_thread_dimensions) ||
	    !ReadFlag(reader, input.needs_lds_barriers) || !reader.U32(input.wave_size) ||
	    !reader.U32(thread_ids_num) || !reader.U32(workgroup_register) ||
	    !ReadFlag(reader, input.tg_size_en) || thread_ids_num > std::numeric_limits<int>::max() ||
	    workgroup_register > std::numeric_limits<int>::max()) {
		return false;
	}
	input.thread_ids_num     = static_cast<int>(thread_ids_num);
	input.workgroup_register = static_cast<int>(workgroup_register);
	return true;
}

bool WritePixelInput(Writer& writer, const ShaderPixelInputInfo& input) {
	if (input.input_num > std::size(input.interpolator_settings) || !writer.U32(input.input_num) ||
	    !writer.U32(input.ps_system_input_base) || !writer.U32(input.custom_interpolation_mask) ||
	    !writer.U32(input.ps_perspective_center_vgpr)) {
		return false;
	}
	for (uint32_t index = 0; index < input.input_num; index++) {
		if (!writer.U32(input.interpolator_settings[index])) return false;
	}
	for (const auto mode: input.target_output_mode) {
		if (!writer.U32(mode)) return false;
	}
	for (const auto mapping: input.target_export_mapping) {
		if (!writer.U32(mapping.packed)) return false;
	}
	return writer.U32(input.ps_pos_x ? 1u : 0u) && writer.U32(input.ps_pos_y ? 1u : 0u) &&
	       writer.U32(input.ps_pos_z ? 1u : 0u) && writer.U32(input.ps_pos_w ? 1u : 0u) &&
	       writer.U32(input.ps_front_face ? 1u : 0u) &&
	       writer.U32(input.ps_no_perspective ? 1u : 0u) &&
	       writer.U32(input.ps_pixel_kill_enable ? 1u : 0u) &&
	       writer.U32(input.ps_depth_export_enable ? 1u : 0u) &&
	       writer.U32(input.ps_sample_mask_export_enable ? 1u : 0u) &&
	       writer.U32(input.ps_early_z ? 1u : 0u);
}

bool ReadPixelInput(Reader& reader, ShaderPixelInputInfo& input) {
	if (!reader.U32(input.input_num) || input.input_num > std::size(input.interpolator_settings) ||
	    !reader.U32(input.ps_system_input_base) || !reader.U32(input.custom_interpolation_mask) ||
	    !reader.U32(input.ps_perspective_center_vgpr)) {
		return false;
	}
	for (uint32_t index = 0; index < input.input_num; index++) {
		if (!reader.U32(input.interpolator_settings[index])) return false;
	}
	for (auto& mode: input.target_output_mode) {
		uint32_t raw = 0;
		if (!reader.U32(raw) || raw > std::numeric_limits<uint8_t>::max()) return false;
		mode = static_cast<uint8_t>(raw);
	}
	for (auto& mapping: input.target_export_mapping) {
		uint32_t raw = 0;
		if (!reader.U32(raw) || raw > std::numeric_limits<uint8_t>::max()) return false;
		mapping.packed = static_cast<uint8_t>(raw);
	}
	return ReadFlag(reader, input.ps_pos_x) && ReadFlag(reader, input.ps_pos_y) &&
	       ReadFlag(reader, input.ps_pos_z) && ReadFlag(reader, input.ps_pos_w) &&
	       ReadFlag(reader, input.ps_front_face) && ReadFlag(reader, input.ps_no_perspective) &&
	       ReadFlag(reader, input.ps_pixel_kill_enable) &&
	       ReadFlag(reader, input.ps_depth_export_enable) &&
	       ReadFlag(reader, input.ps_sample_mask_export_enable) &&
	       ReadFlag(reader, input.ps_early_z);
}

bool WriteResources(Writer& writer, const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	if (!WriteDescriptors(writer, resources.buffers) ||
	    !WriteDescriptors(writer, resources.images) ||
	    !WriteDescriptors(writer, resources.samplers) ||
	    !WriteWords(writer, resources.flattened_srt) || !WriteWords(writer, resources.user_data) ||
	    resources.indirect_images.size() > std::numeric_limits<uint32_t>::max() ||
	    !writer.U32(static_cast<uint32_t>(resources.indirect_images.size()))) {
		return false;
	}
	for (const auto& table: resources.indirect_images) {
		if (!writer.U32(table.resource) || !writer.U32(table.capacity) ||
		    !WriteWords(writer, table.keys) || !WriteWords(writer, table.candidates) ||
		    !WriteDescriptors(writer, table.descriptors)) {
			return false;
		}
	}
	return true;
}

bool ReadResources(Reader& reader, ShaderRecompiler::IR::ResourceSnapshot& resources) {
	if (!ReadDescriptors(reader, resources.buffers) || !ReadDescriptors(reader, resources.images) ||
	    !ReadDescriptors(reader, resources.samplers) ||
	    !ReadWords(reader, resources.flattened_srt, MaxCaptureWords) ||
	    !ReadWords(reader, resources.user_data, 64u)) {
		return false;
	}
	uint32_t count = 0;
	if (!reader.U32(count) || count > MaxCaptureDescriptors) return false;
	resources.indirect_images.resize(count);
	for (auto& table: resources.indirect_images) {
		if (!reader.U32(table.resource) || !reader.U32(table.capacity) ||
		    !ReadWords(reader, table.keys, MaxCaptureDescriptors) ||
		    !ReadWords(reader, table.candidates, MaxCaptureDescriptors) ||
		    !ReadDescriptors(reader, table.descriptors) ||
		    table.keys.size() != table.candidates.size() || table.keys.size() > table.capacity) {
			return false;
		}
	}
	return true;
}

} // namespace

bool WriteComputeCapture(const std::filesystem::path& file, std::span<const uint32_t> code,
                         const ShaderRecompiler::CompileOptions&       options,
                         const ShaderComputeInputInfo&                 input,
                         const ShaderRecompiler::IR::ResourceSnapshot& resources,
                         std::string*                                  error) {
	if (options.stage != ShaderType::Compute || options.user_data_count > 64u ||
	    (options.user_data_count != 0u && options.user_data == nullptr) ||
	    input.thread_ids_num < 0 || input.workgroup_register < 0) {
		return Fail(error, "invalid compute shader replay inputs");
	}
	Writer writer(file);
	if (!writer.Valid())
		return Fail(error, "could not create compute shader replay: " + file.string());
	const std::span<const uint32_t> user_data(options.user_data, options.user_data_count);
	if (!writer.Bytes(CaptureMagic.data(), CaptureMagic.size()) || !writer.U32(CaptureVersion) ||
	    !writer.U64(options.shader_hash) || !writer.U64(options.shader_base) ||
	    !writer.U32(options.user_data_base) || !writer.U32(options.user_data_count) ||
	    !writer.U32(options.scratch_dwords) || !writer.U32(options.push_constant_offset) ||
	    !WriteInput(writer, input) || !WriteWords(writer, code) || !WriteWords(writer, user_data) ||
	    !WriteResources(writer, resources)) {
		return Fail(error, "could not write compute shader replay: " + file.string());
	}
	return true;
}

bool ReadComputeCapture(const std::filesystem::path& file, ComputeCapture& capture,
                        std::string* error) {
	Reader reader(file);
	if (!reader.Valid())
		return Fail(error, "could not open compute shader replay: " + file.string());
	std::array<char, CaptureMagic.size()> magic {};
	uint32_t                              version = 0;
	ComputeCapture                        next;
	if (!reader.Bytes(magic.data(), magic.size()) || magic != CaptureMagic ||
	    !reader.U32(version) || version != CaptureVersion || !reader.U64(next.shader_hash) ||
	    !reader.U64(next.shader_base) || !reader.U32(next.user_data_base) ||
	    !reader.U32(next.user_data_count) || !reader.U32(next.scratch_dwords) ||
	    !reader.U32(next.push_constant_offset) || !ReadInput(reader, next.input) ||
	    !ReadWords(reader, next.code, MaxCaptureWords) || !ReadWords(reader, next.user_data, 64u) ||
	    !ReadResources(reader, next.resources) || !reader.AtEnd()) {
		return Fail(error, "invalid or truncated compute shader replay: " + file.string());
	}
	if (next.code.empty() || next.user_data_count != next.user_data.size() ||
	    (next.input.wave_size != 32u && next.input.wave_size != 64u)) {
		return Fail(error, "compute shader replay has inconsistent ABI sizes");
	}
	capture = std::move(next);
	return true;
}

bool WritePixelCapture(const std::filesystem::path& file, std::span<const uint32_t> code,
                       const ShaderRecompiler::CompileOptions&       options,
                       const ShaderPixelInputInfo&                   input,
                       const ShaderRecompiler::IR::ResourceSnapshot& resources,
                       std::string*                                  error) {
	if (options.stage != ShaderType::Pixel ||
	    (options.wave_size != 32u && options.wave_size != 64u) || options.user_data_count > 64u ||
	    (options.user_data_count != 0u && options.user_data == nullptr) ||
	    input.input_num > std::size(input.interpolator_settings)) {
		return Fail(error, "invalid pixel shader replay inputs");
	}
	Writer writer(file);
	if (!writer.Valid())
		return Fail(error, "could not create pixel shader replay: " + file.string());
	const std::span<const uint32_t> user_data(options.user_data, options.user_data_count);
	if (!writer.Bytes(PixelCaptureMagic.data(), PixelCaptureMagic.size()) ||
	    !writer.U32(CaptureVersion) || !writer.U64(options.shader_hash) ||
	    !writer.U64(options.shader_base) || !writer.U32(options.wave_size) ||
	    !writer.U32(options.user_data_base) || !writer.U32(options.user_data_count) ||
	    !writer.U32(options.scratch_dwords) || !writer.U32(options.push_constant_offset) ||
	    !WritePixelInput(writer, input) || !WriteWords(writer, code) ||
	    !WriteWords(writer, user_data) || !WriteResources(writer, resources)) {
		return Fail(error, "could not write pixel shader replay: " + file.string());
	}
	return true;
}

bool ReadPixelCapture(const std::filesystem::path& file, PixelCapture& capture,
                      std::string* error) {
	Reader reader(file);
	if (!reader.Valid()) return Fail(error, "could not open pixel shader replay: " + file.string());
	std::array<char, PixelCaptureMagic.size()> magic {};
	uint32_t                                   version = 0;
	PixelCapture                               next;
	if (!reader.Bytes(magic.data(), magic.size()) || magic != PixelCaptureMagic ||
	    !reader.U32(version) || version != CaptureVersion || !reader.U64(next.shader_hash) ||
	    !reader.U64(next.shader_base) || !reader.U32(next.wave_size) ||
	    !reader.U32(next.user_data_base) || !reader.U32(next.user_data_count) ||
	    !reader.U32(next.scratch_dwords) || !reader.U32(next.push_constant_offset) ||
	    !ReadPixelInput(reader, next.input) || !ReadWords(reader, next.code, MaxCaptureWords) ||
	    !ReadWords(reader, next.user_data, 64u) || !ReadResources(reader, next.resources) ||
	    !reader.AtEnd()) {
		return Fail(error, "invalid or truncated pixel shader replay: " + file.string());
	}
	if (next.code.empty() || next.user_data_count != next.user_data.size() ||
	    (next.wave_size != 32u && next.wave_size != 64u)) {
		return Fail(error, "pixel shader replay has inconsistent ABI sizes");
	}
	next.input.push_constant_offset = next.push_constant_offset;
	next.input.scratch_size_dwords  = next.scratch_dwords;
	capture                         = std::move(next);
	return true;
}

} // namespace Libs::Graphics::ShaderReplay
