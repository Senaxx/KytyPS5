#include "graphics/shader/shaderReplay.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "spirv-tools/libspirv.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics {
namespace {

bool WriteBinary(const std::filesystem::path &file,
                 std::span<const uint32_t> words, std::string &error) {
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "could not create " + file.string();
    return false;
  }
  out.write(reinterpret_cast<const char *>(words.data()),
            static_cast<std::streamsize>(words.size_bytes()));
  if (!out) {
    error = "could not write " + file.string();
    return false;
  }
  return true;
}

bool WriteText(const std::filesystem::path &file, const std::string &text,
               std::string &error) {
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "could not create " + file.string();
    return false;
  }
  out << text;
  if (!out) {
    error = "could not write " + file.string();
    return false;
  }
  return true;
}

bool ValidateSpirv(std::span<const uint32_t> spirv, std::string &text,
                   std::string &error) {
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
  tools.SetMessageConsumer([&error](spv_message_level_t, const char *,
                                    const spv_position_t &pos,
                                    const char *message) {
    error += std::to_string(pos.index) + ": " + message + '\n';
  });
  const bool valid = tools.Validate(spirv.data(), spirv.size());
  const bool disassembled =
      tools.Disassemble(spirv.data(), spirv.size(), &text);
  return valid && disassembled;
}

void InitializeConfig(Common::Subsystems &subsystems) {
  Common::InitializeThreads();
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions config;
  config.printf_direction = Config::OutputDirection::Console;
  config.shader_log_direction = Config::ShaderLogDirection::Silent;
  Config::Load(config);
  subsystems.Initialize<Log::Lifecycle>();
}

void PrintIndirectImages(
    const ShaderRecompiler::IR::ResourceSnapshot &resources) {
  for (size_t table_index = 0; table_index < resources.indirect_images.size();
       ++table_index) {
    const auto &table = resources.indirect_images[table_index];
    std::printf("indirect_image[%zu]: resource=%u capacity=%u entries=%zu\n",
                table_index, table.resource, table.capacity, table.keys.size());
    for (size_t entry_index = 0; entry_index < table.keys.size();
         ++entry_index) {
      std::printf("  [%zu] key=0x%08x candidate=%u descriptor=", entry_index,
                  table.keys[entry_index], table.candidates[entry_index]);
      const auto candidate = table.candidates[entry_index];
      if (candidate >= table.descriptors.size()) {
        std::printf("<none>\n");
        continue;
      }
      const auto &descriptor = table.descriptors[candidate];
      for (uint32_t word = 0; word < descriptor.dword_count; ++word) {
        std::printf("%s%08x", word == 0 ? "" : ":", descriptor.dwords[word]);
      }
      std::printf("\n");
    }
  }
}

void PrintDescriptors(
    std::string_view label,
    std::span<const ShaderRecompiler::IR::DescriptorValue> descriptors) {
  for (size_t index = 0; index < descriptors.size(); ++index) {
    const auto &descriptor = descriptors[index];
    std::printf("%.*s[%zu]=", static_cast<int>(label.size()), label.data(),
                index);
    for (uint32_t word = 0; word < descriptor.dword_count; ++word) {
      std::printf("%s%08x", word == 0 ? "" : ":", descriptor.dwords[word]);
    }
    std::printf("\n");
  }
}

int SelfTest() {
  ShaderComputeInputInfo input;
  input.threads_num[0] = 8;
  input.threads_num[1] = 8;
  input.threads_num[2] = 1;
  input.lds_size_dwords = 128;
  input.group_id[0] = true;
  input.needs_lds_barriers = true;
  input.host_subgroup_size = 32;
  input.wave_size = 64;
  input.thread_ids_num = 2;
  input.workgroup_register = 40;
  std::vector<uint32_t> code = {0xbe0003ffu, 0xbf810000u};
  std::vector<uint32_t> user_data = {1u, 2u, 3u, 4u};
  ShaderRecompiler::IR::ResourceSnapshot resources;
  resources.user_data = user_data;
  resources.flattened_srt = {0x10u, 0x20u};
  ShaderRecompiler::IR::DescriptorValue image;
  image.dword_count = 8;
  image.dwords = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
  resources.images.push_back(image);
  ShaderRecompiler::IR::ResourceSnapshot::IndirectImage table;
  table.resource = 0;
  table.capacity = 4;
  table.keys = {7u, 9u};
  table.candidates = {0u, 1u};
  table.descriptors = {image, image};
  resources.indirect_images.push_back(table);
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.shader_hash = 0xb3ee9eea5b92b870ull;
  options.shader_base = 0x123456780000ull;
  options.user_data_count = static_cast<uint32_t>(user_data.size());
  options.user_data = user_data.data();
  options.scratch_dwords = 32;
  options.input_info.compute = &input;

  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto file = std::filesystem::temp_directory_path() /
                    ("kyty_shader_replay_" + std::to_string(stamp) + ".replay");
  std::string error;
  if (!ShaderReplay::WriteComputeCapture(file, code, options, input, resources,
                                         &error)) {
    std::fprintf(stderr, "shader_replay self-test: %s\n", error.c_str());
    return 1;
  }
  ShaderReplay::ComputeCapture capture;
  const bool read = ShaderReplay::ReadComputeCapture(file, capture, &error);
  std::error_code remove_error;
  std::filesystem::remove(file, remove_error);
  if (!read || capture.shader_hash != options.shader_hash ||
      capture.code != code || capture.user_data != user_data ||
      capture.input.threads_num[0] != 8u ||
      capture.input.needs_lds_barriers != input.needs_lds_barriers ||
      capture.input.host_subgroup_size != input.host_subgroup_size ||
      capture.resources.images != resources.images ||
      capture.resources.indirect_images.size() != 1u ||
      capture.resources.indirect_images[0].keys != table.keys ||
      capture.resources.indirect_images[0].candidates != table.candidates ||
      capture.resources.indirect_images[0].descriptors != table.descriptors) {
    std::fprintf(stderr, "shader_replay self-test: %s\n",
                 error.empty() ? "round-trip mismatch" : error.c_str());
    return 1;
  }

  ShaderPixelInputInfo pixel_input;
  pixel_input.input_num = 2;
  pixel_input.interpolator_settings[0] = 3u;
  pixel_input.interpolator_settings[1] = 0x401u;
  pixel_input.ps_system_input_base = 9u;
  pixel_input.custom_interpolation_mask = 1u;
  pixel_input.ps_perspective_center_vgpr = 4u;
  pixel_input.target_output_mode[0] = 4u;
  pixel_input.target_export_mapping[0].packed = 0xc6u;
  pixel_input.push_constant_offset = 32u;
  pixel_input.scratch_size_dwords = 16u;
  pixel_input.ps_pos_x = true;
  pixel_input.ps_pos_z = true;
  pixel_input.ps_front_face = true;
  pixel_input.ps_no_perspective = true;
  pixel_input.ps_pixel_kill_enable = true;
  pixel_input.ps_depth_export_enable = true;
  pixel_input.ps_sample_mask_export_enable = true;
  pixel_input.ps_early_z = true;
  ShaderRecompiler::CompileOptions pixel_options;
  pixel_options.stage = ShaderType::Pixel;
  pixel_options.wave_size = 32u;
  pixel_options.shader_hash = 0x0000000036e18a55ull;
  pixel_options.shader_base = 0x123456790000ull;
  pixel_options.user_data_base = 2u;
  pixel_options.user_data_count = static_cast<uint32_t>(user_data.size());
  pixel_options.user_data = user_data.data();
  pixel_options.scratch_dwords = pixel_input.scratch_size_dwords;
  pixel_options.push_constant_offset = pixel_input.push_constant_offset;
  pixel_options.input_info.pixel = &pixel_input;
  const auto pixel_file =
      std::filesystem::temp_directory_path() /
      ("kyty_pixel_shader_replay_" + std::to_string(stamp) + ".replay");
  if (!ShaderReplay::WritePixelCapture(pixel_file, code, pixel_options,
                                       pixel_input, resources, &error)) {
    std::fprintf(stderr, "shader_replay pixel self-test: %s\n", error.c_str());
    return 1;
  }
  ShaderReplay::PixelCapture pixel_capture;
  error.clear();
  const bool pixel_read =
      ShaderReplay::ReadPixelCapture(pixel_file, pixel_capture, &error);
  std::filesystem::remove(pixel_file, remove_error);
  if (!pixel_read || pixel_capture.shader_hash != pixel_options.shader_hash ||
      pixel_capture.shader_base != pixel_options.shader_base ||
      pixel_capture.wave_size != pixel_options.wave_size ||
      pixel_capture.user_data_base != pixel_options.user_data_base ||
      pixel_capture.code != code || pixel_capture.user_data != user_data ||
      pixel_capture.input.input_num != pixel_input.input_num ||
      pixel_capture.input.interpolator_settings[0] !=
          pixel_input.interpolator_settings[0] ||
      pixel_capture.input.interpolator_settings[1] !=
          pixel_input.interpolator_settings[1] ||
      pixel_capture.input.ps_system_input_base !=
          pixel_input.ps_system_input_base ||
      pixel_capture.input.custom_interpolation_mask !=
          pixel_input.custom_interpolation_mask ||
      pixel_capture.input.ps_perspective_center_vgpr !=
          pixel_input.ps_perspective_center_vgpr ||
      pixel_capture.input.target_output_mode[0] !=
          pixel_input.target_output_mode[0] ||
      pixel_capture.input.target_export_mapping[0] !=
          pixel_input.target_export_mapping[0] ||
      pixel_capture.input.push_constant_offset !=
          pixel_input.push_constant_offset ||
      pixel_capture.input.scratch_size_dwords !=
          pixel_input.scratch_size_dwords ||
      pixel_capture.input.ps_pos_x != pixel_input.ps_pos_x ||
      pixel_capture.input.ps_pos_z != pixel_input.ps_pos_z ||
      pixel_capture.input.ps_front_face != pixel_input.ps_front_face ||
      pixel_capture.input.ps_no_perspective != pixel_input.ps_no_perspective ||
      pixel_capture.input.ps_pixel_kill_enable !=
          pixel_input.ps_pixel_kill_enable ||
      pixel_capture.input.ps_depth_export_enable !=
          pixel_input.ps_depth_export_enable ||
      pixel_capture.input.ps_sample_mask_export_enable !=
          pixel_input.ps_sample_mask_export_enable ||
      pixel_capture.input.ps_early_z != pixel_input.ps_early_z ||
      pixel_capture.resources.images != resources.images) {
    std::fprintf(stderr, "shader_replay pixel self-test: %s\n",
                 error.empty() ? "round-trip mismatch" : error.c_str());
    return 1;
  }

  capture.code = {0xbf810000u};
  capture.user_data.clear();
  capture.user_data_count = 0;
  capture.resources = {};
  Common::Subsystems subsystems;
  InitializeConfig(subsystems);

  const std::vector<uint32_t> incomplete_code = {
      0xe0702008u, // buffer_store_dword v0, v1, s[0:3], 0 offset:8
      0x80000001u, 0xbf810000u};
  ShaderRecompiler::IR::ResourceSnapshot incomplete_resources;
  ShaderRecompiler::CompileOptions incomplete_options;
  incomplete_options.stage = ShaderType::Compute;
  incomplete_options.dump_ir = true;
  incomplete_options.resource_snapshot = &incomplete_resources;
  incomplete_options.input_info.compute = &capture.input;
  ShaderRecompiler::CompileResult incomplete_result;
  error.clear();
  if (ShaderRecompiler::TryRecompile(incomplete_code, incomplete_options,
                                     incomplete_result, &error) ||
      incomplete_result.decoded_dump.empty() ||
      incomplete_result.ir_dump.empty() ||
      error.find("resource snapshot") == std::string::npos) {
    std::fprintf(stderr,
                 "shader_replay incomplete-resource self-test failed: %s\n",
                 error.c_str());
    return 1;
  }

  ShaderRecompiler::CompileOptions replay_options;
  replay_options.stage = ShaderType::Compute;
  replay_options.wave_size = capture.input.wave_size;
  replay_options.shader_hash = capture.shader_hash;
  replay_options.user_data_count = 0;
  replay_options.dump_ir = true;
  replay_options.resource_snapshot = &capture.resources;
  replay_options.input_info.compute = &capture.input;
  ShaderRecompiler::CompileResult replay_result;
  if (!ShaderRecompiler::TryRecompile(capture.code, replay_options,
                                      replay_result, &error) ||
      replay_result.decoded_dump.empty() || replay_result.ir_dump.empty() ||
      replay_result.spirv.empty()) {
    std::fprintf(stderr, "shader_replay self-test: replay failed: %s\n",
                 error.c_str());
    return 1;
  }

  pixel_capture.code = {0xbf810000u};
  pixel_capture.user_data.clear();
  pixel_capture.user_data_count = 0;
  pixel_capture.resources = {};
  ShaderRecompiler::CompileOptions pixel_replay_options;
  pixel_replay_options.stage = ShaderType::Pixel;
  pixel_replay_options.wave_size = pixel_capture.wave_size;
  pixel_replay_options.shader_hash = pixel_capture.shader_hash;
  pixel_replay_options.user_data_base = pixel_capture.user_data_base;
  pixel_replay_options.user_data_count = 0;
  pixel_replay_options.scratch_dwords = pixel_capture.scratch_dwords;
  pixel_replay_options.push_constant_offset =
      pixel_capture.push_constant_offset;
  pixel_replay_options.dump_ir = true;
  pixel_replay_options.resource_snapshot = &pixel_capture.resources;
  pixel_replay_options.input_info.pixel = &pixel_capture.input;
  ShaderRecompiler::CompileResult pixel_replay_result;
  error.clear();
  if (!ShaderRecompiler::TryRecompile(pixel_capture.code, pixel_replay_options,
                                      pixel_replay_result, &error) ||
      pixel_replay_result.decoded_dump.empty() ||
      pixel_replay_result.ir_dump.empty() ||
      pixel_replay_result.spirv.empty()) {
    std::fprintf(stderr, "shader_replay pixel self-test: replay failed: %s\n",
                 error.c_str());
    return 1;
  }
  std::string pixel_spirv_text;
  error.clear();
  if (!ValidateSpirv(pixel_replay_result.spirv, pixel_spirv_text, error)) {
    std::fprintf(
        stderr, "shader_replay pixel self-test: SPIR-V validation failed: %s\n",
        error.c_str());
    return 1;
  }
  std::printf("shader_replay self-test: ok\n");
  return 0;
}

} // namespace
} // namespace Libs::Graphics

int main(int argc, char **argv) {
  using namespace Libs::Graphics;
  if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
    return SelfTest();
  }
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr,
                 "Usage: shader_replay <capture.replay> [output-prefix]\n");
    return 1;
  }

  ShaderReplay::ComputeCapture compute_capture;
  ShaderReplay::PixelCapture pixel_capture;
  std::string error;
  bool pixel = false;
  if (!ShaderReplay::ReadComputeCapture(argv[1], compute_capture, &error)) {
    error.clear();
    if (!ShaderReplay::ReadPixelCapture(argv[1], pixel_capture, &error)) {
      std::fprintf(stderr, "shader_replay: %s\n", error.c_str());
      return 1;
    }
    pixel = true;
  }
  const auto &resources =
      pixel ? pixel_capture.resources : compute_capture.resources;
  const auto &code = pixel ? pixel_capture.code : compute_capture.code;
  const auto &user_data =
      pixel ? pixel_capture.user_data : compute_capture.user_data;
  const auto shader_hash =
      pixel ? pixel_capture.shader_hash : compute_capture.shader_hash;
  PrintIndirectImages(resources);
  PrintDescriptors("buffer", resources.buffers);
  PrintDescriptors("image", resources.images);
  PrintDescriptors("sampler", resources.samplers);
  Common::Subsystems subsystems;
  InitializeConfig(subsystems);
  ShaderRecompiler::CompileOptions options;
  options.stage = pixel ? ShaderType::Pixel : ShaderType::Compute;
  options.wave_size =
      pixel ? pixel_capture.wave_size : compute_capture.input.wave_size;
  options.user_data_base =
      pixel ? pixel_capture.user_data_base : compute_capture.user_data_base;
  options.user_data_count =
      pixel ? pixel_capture.user_data_count : compute_capture.user_data_count;
  options.scratch_dwords =
      pixel ? pixel_capture.scratch_dwords : compute_capture.scratch_dwords;
  options.shader_hash = shader_hash;
  options.shader_base =
      pixel ? pixel_capture.shader_base : compute_capture.shader_base;
  options.push_constant_offset = pixel ? pixel_capture.push_constant_offset
                                       : compute_capture.push_constant_offset;
  options.dump_ir = true;
  options.dump_label = "ShaderReplay";
  options.user_data = user_data.data();
  options.resource_snapshot = &resources;
  if (pixel) {
    options.input_info.pixel = &pixel_capture.input;
  } else {
    options.input_info.compute = &compute_capture.input;
  }

  ShaderRecompiler::CompileResult result;
  if (!ShaderRecompiler::TryRecompile(code, options, result, &error)) {
    if (argc == 3) {
      auto decoded = std::filesystem::path(argv[2]);
      decoded += ".rdna2";
      auto ir = std::filesystem::path(argv[2]);
      ir += ".ir";
      std::string write_error;
      WriteText(decoded, result.decoded_dump, write_error);
      WriteText(ir, result.ir_dump, write_error);
    }
    std::fprintf(stderr, "shader_replay: compile failed: %s\n", error.c_str());
    return 2;
  }
  std::string spirv_text;
  error.clear();
  const bool spirv_valid = ValidateSpirv(result.spirv, spirv_text, error);
  const auto validation_error = error;

  std::filesystem::path output = argc == 3 ? argv[2] : argv[1];
  if (argc != 3)
    output.replace_extension();
  auto decoded = output;
  decoded += ".rdna2";
  auto ir = output;
  ir += ".ir";
  auto spv = output;
  spv += ".spv";
  auto spvasm = output;
  spvasm += ".spvasm";
  if (!WriteText(decoded, result.decoded_dump, error) ||
      !WriteText(ir, result.ir_dump, error) ||
      !WriteBinary(spv, result.spirv, error) ||
      !WriteText(spvasm, spirv_text, error)) {
    std::fprintf(stderr, "shader_replay: %s\n", error.c_str());
    return 4;
  }
  if (!spirv_valid) {
    std::fprintf(stderr,
                 "shader_replay: SPIR-V validation/disassembly failed: %s\n",
                 validation_error.c_str());
    return 3;
  }
  std::printf("shader_replay: hash=0x%016llx code=%zu SPIR-V=%zu words\n",
              static_cast<unsigned long long>(shader_hash), code.size(),
              result.spirv.size());
  return 0;
}
