#include "mainWindow.h"

#include <QComboBox>
#include <QCheckBox>
#include <QDir>
#include <QFile>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QLocalSocket>
#include <QPlainTextEdit>
#include <QPainter>
#include <QPushButton>
#include <QScrollBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QSplitter>
#include <QPixmap>
#include <QWidget>

#include <algorithm>
#include <limits>

#if defined(Q_OS_WIN)
#include <windows.h>
#endif

class ResourceAliasMapWidget final: public QWidget {
public:
	struct Lane {
		QString label;
		QString owner;
		quint64 address = 0;
		quint64 size = 0;
		quint64 stencil_address = 0;
		quint64 stencil_size = 0;
		quint64 metadata_address = 0;
		quint64 metadata_size = 0;
		bool active = false;
	};

	explicit ResourceAliasMapWidget(QWidget* parent = nullptr): QWidget(parent) {
		setMinimumHeight(150);
		setToolTip("Blue: pixel data   Purple: stencil   Orange: compression metadata");
	}

	void SetAliases(QVector<Lane> lanes) {
		m_lanes = std::move(lanes);
		update();
	}

protected:
	void paintEvent(QPaintEvent*) override {
		QPainter painter(this);
		painter.fillRect(rect(), palette().brush(QPalette::Base));
		if (m_lanes.isEmpty()) {
			painter.setPen(palette().color(QPalette::Disabled, QPalette::Text));
			painter.drawText(rect(), Qt::AlignCenter,
			                 "Trace a range to visualize overlapping pixel, stencil and metadata storage");
			return;
		}

		quint64 begin = (std::numeric_limits<quint64>::max)();
		quint64 end = 0;
		const auto include = [&begin, &end](quint64 address, quint64 size) {
			if (address == 0 || size == 0) return;
			begin = (std::min)(begin, address);
			end = (std::max)(end, size > (std::numeric_limits<quint64>::max)() - address
			                        ? (std::numeric_limits<quint64>::max)()
			                        : address + size);
		};
		for (const auto& lane: m_lanes) {
			include(lane.address, lane.size);
			include(lane.stencil_address, lane.stencil_size);
			include(lane.metadata_address, lane.metadata_size);
		}
		if (begin == (std::numeric_limits<quint64>::max)() || end <= begin) return;

		constexpr int label_width = 145;
		const int plot_left = label_width;
		const int plot_right = (std::max)(plot_left + 1, width() - 10);
		const int shown = (std::min)(static_cast<int>(m_lanes.size()), 12);
		const int lane_height = (std::max)(10, (height() - 34) / (std::max)(1, shown));
		const auto x = [=](quint64 address) {
			const long double ratio = static_cast<long double>(address - begin) /
			                          static_cast<long double>(end - begin);
			return plot_left + static_cast<int>(ratio * (plot_right - plot_left));
		};
		const auto bar = [&](const Lane& lane, quint64 address, quint64 size, int y,
		                     QColor color) {
			if (address == 0 || size == 0) return;
			const int left = x(address);
			const quint64 range_end = size > (std::numeric_limits<quint64>::max)() - address
			                              ? (std::numeric_limits<quint64>::max)()
			                              : address + size;
			const int right = (std::max)(left + 2, x(range_end));
			if (!lane.active) color.setAlpha(80);
			painter.fillRect(QRect(left, y, right - left, 7), color);
			if (lane.owner == "native image") {
				painter.setPen(QColor(80, 220, 120));
				painter.drawRect(QRect(left, y, right - left, 7));
			}
		};

		painter.setPen(palette().color(QPalette::Text));
		painter.drawText(4, 15, QString("0x%1 — 0x%2").arg(begin, 0, 16).arg(end, 0, 16));
		painter.setPen(QColor(70, 140, 240)); painter.drawText(width() - 290, 15, "pixel");
		painter.setPen(QColor(170, 100, 230)); painter.drawText(width() - 225, 15, "stencil");
		painter.setPen(QColor(240, 150, 50)); painter.drawText(width() - 145, 15, "metadata");
		for (int index = 0; index < shown; index++) {
			const auto& lane = m_lanes[index];
			const int y = 25 + index * lane_height;
			painter.setPen(lane.active ? palette().color(QPalette::Text)
			                           : palette().color(QPalette::Disabled, QPalette::Text));
			painter.drawText(4, y + 8, lane.label.left(22));
			bar(lane, lane.address, lane.size, y, QColor(70, 140, 240));
			bar(lane, lane.stencil_address, lane.stencil_size, y, QColor(170, 100, 230));
			bar(lane, lane.metadata_address, lane.metadata_size, y, QColor(240, 150, 50));
		}
		if (m_lanes.size() > shown) {
			painter.setPen(palette().color(QPalette::Disabled, QPalette::Text));
			painter.drawText(4, height() - 3,
			                 QString("+ %1 older aliases in the table").arg(m_lanes.size() - shown));
		}
	}

private:
	QVector<Lane> m_lanes;
};

namespace {

class SortableTableItem final: public QTableWidgetItem {
public:
	explicit SortableTableItem(const QString& text): QTableWidgetItem(text) {
		auto number = text.trimmed();
		if (number.endsWith(" ms")) {
			number.chop(3);
			m_decimal = number.toDouble(&m_numeric);
			m_is_decimal = m_numeric;
			return;
		}
		if (number.startsWith("0x", Qt::CaseInsensitive)) {
			m_unsigned = number.mid(2).toULongLong(&m_numeric, 16);
			return;
		}
		m_signed = number.toLongLong(&m_numeric, 10);
	}

	bool operator<(const QTableWidgetItem& other) const override {
		const auto* sortable = dynamic_cast<const SortableTableItem*>(&other);
		if (sortable != nullptr && m_numeric && sortable->m_numeric) {
			if (m_is_decimal || sortable->m_is_decimal) {
				const auto left = m_is_decimal ? m_decimal : static_cast<double>(m_signed);
				const auto right = sortable->m_is_decimal ? sortable->m_decimal
				                                             : static_cast<double>(sortable->m_signed);
				return left < right;
			}
			if (text().startsWith("0x", Qt::CaseInsensitive) &&
			    other.text().startsWith("0x", Qt::CaseInsensitive))
				return m_unsigned < sortable->m_unsigned;
			return m_signed < sortable->m_signed;
		}
		return QString::localeAwareCompare(text().toLower(), other.text().toLower()) < 0;
	}

private:
	bool    m_numeric = false;
	bool    m_is_decimal = false;
	quint64 m_unsigned = 0;
	qint64  m_signed = 0;
	double  m_decimal = 0.0;
};

class TableUpdateGuard final {
public:
	explicit TableUpdateGuard(QTableWidget* table): m_table(table), m_sorting(table->isSortingEnabled()),
	                                                    m_signals_blocked(table->blockSignals(true)) {
		m_table->setSortingEnabled(false);
	}
	~TableUpdateGuard() {
		m_table->setSortingEnabled(m_sorting);
		m_table->blockSignals(m_signals_blocked);
	}

private:
	QTableWidget* m_table;
	bool          m_sorting;
	bool          m_signals_blocked;
};

QString Hex(quint64 value) {
	return QString("0x%1").arg(value, 0, 16);
}

QTableWidget* MakeTable(const QStringList& headers) {
	auto* table = new QTableWidget();
	table->setColumnCount(headers.size());
	table->setHorizontalHeaderLabels(headers);
	table->setEditTriggers(QAbstractItemView::NoEditTriggers);
	table->setSelectionBehavior(QAbstractItemView::SelectRows);
	table->setAlternatingRowColors(true);
	table->setSortingEnabled(true);
	table->horizontalHeader()->setSortIndicatorShown(true);
	table->horizontalHeader()->setStretchLastSection(true);
	return table;
}

void SetItem(QTableWidget* table, int row, int column, const QString& text) {
	table->setItem(row, column, new SortableTableItem(text));
}

quint64 JsonU64(const QJsonValue& value) {
	return value.toVariant().toULongLong();
}

QString ResourceIdentity(const QJsonObject& value) {
	return value.value("has_image").toBool()
	           ? QString("%1:%2").arg(value.value("image_index").toInt())
	                 .arg(value.value("image_generation").toInt())
	           : QString();
}

QString ResourceOwner(const QJsonObject& value) {
	if (value.value("gpu_modified").toBool()) return "native image";
	if (value.value("buffer_modified").toBool()) return "buffer cache";
	if (value.value("cpu_dirty").toBool()) return "guest memory";
	if (value.value("maybe_cpu_dirty").toBool()) return "guest page?";
	return "coherent";
}

QString ResourceDirty(const QJsonObject& value) {
	QStringList state;
	if (value.value("gpu_modified").toBool()) state << "GPU";
	if (value.value("buffer_modified").toBool()) state << "BUF";
	if (value.value("cpu_dirty").toBool()) state << "CPU";
	if (value.value("maybe_cpu_dirty").toBool()) state << "?CPU";
	return state.isEmpty() ? "clean" : state.join('|');
}

QString ResourceUsage(const QJsonObject& value) {
	QStringList usage;
	if (value.value("usage_texture").toBool()) usage << "texture";
	if (value.value("usage_storage").toBool()) usage << "storage";
	if (value.value("usage_render_target").toBool()) usage << "color target";
	if (value.value("usage_depth_target").toBool()) usage << "depth target";
	if (value.value("usage_video_out").toBool()) usage << "video out";
	return usage.join(", ");
}

QString ResourceShader(const QJsonObject& value) {
	const auto cs = JsonU64(value.value("cs"));
	if (cs != 0) return "CS " + Hex(cs);
	QStringList shaders;
	const auto vs = JsonU64(value.value("vs"));
	const auto ps = JsonU64(value.value("ps"));
	if (vs != 0) shaders << "VS " + Hex(vs);
	if (ps != 0) shaders << "PS " + Hex(ps);
	return shaders.join(" / ");
}

bool ProcessIsRunning(quint64 pid) {
#if defined(Q_OS_WIN)
	const auto process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
	if (process == nullptr) return false;
	const bool running = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
	CloseHandle(process);
	return running;
#else
	(void)pid;
	return true;
#endif
}

} // namespace

MainWindow::MainWindow(QWidget* parent): QMainWindow(parent) {
	setWindowTitle("Kyty External Debugger");

	auto* central = new QWidget(this);
	auto* layout = new QVBoxLayout(central);
	auto* toolbar = new QHBoxLayout();
	m_sessions = new QComboBox();
	auto* refresh = new QPushButton("Refresh sessions");
	m_connect = new QPushButton("Attach");
	m_pause = new QPushButton("Pause");
	m_continue = new QPushButton("Continue");
	m_step_into = new QPushButton("Step Into");
	m_step_over = new QPushButton("Step Over");
	m_step_out = new QPushButton("Step Out");
	m_status = new QLabel("Not attached");
	m_summary = new QLabel("No emulator session selected.");
	m_summary->setTextInteractionFlags(Qt::TextSelectableByMouse);

	toolbar->addWidget(m_sessions, 1);
	toolbar->addWidget(refresh);
	toolbar->addWidget(m_connect);
	toolbar->addSpacing(12);
	toolbar->addWidget(m_pause);
	toolbar->addWidget(m_continue);
	toolbar->addWidget(m_step_into);
	toolbar->addWidget(m_step_over);
	toolbar->addWidget(m_step_out);
	toolbar->addSpacing(12);
	toolbar->addWidget(m_status);
	layout->addLayout(toolbar);
	layout->addWidget(m_summary);

	auto* tabs = new QTabWidget();
	m_threads = MakeTable({"ID", "Guest ID", "Host ID", "Name", "State", "Address"});
	m_modules = MakeTable({"ID", "Module", "Base", "Size"});
	m_symbols = MakeTable({"Symbol", "Module", "Address"});
	m_shaders = MakeTable({"Sequence", "Stage", "Hash", "Guest address", "GCN bytes", "SPIR-V words",
	                       "Resources"});
	m_frame = MakeTable({"#", "Kind", "Submit", "Count / groups", "VS", "PS", "CS"});
	m_registers = MakeTable({"Register", "Value"});
	m_callstack = MakeTable({"#", "Address", "Description"});
	m_disassembly = MakeTable({"Address", "Bytes", "Instruction"});
	m_breakpoints = MakeTable({"ID", "Location", "State", "Hits"});
	m_resources = MakeTable({"Kind", "#", "Access", "Address", "Extent", "Format / tile",
	                         "Source", "First use", "Descriptor"});
	m_resource_history = MakeTable({"Sequence", "Frame", "Action", "Address", "Size", "Extent", "BPE", "Formats", "Tile"});
	m_resource_trace_aliases = MakeTable({"State", "Image", "Guest range", "Host image", "Owner",
	                                      "Dirty", "Usage", "Extent", "Format / tile",
	                                      "Samples / mips / layers", "Stencil", "Metadata",
	                                      "Last event"});
	m_resource_trace_events = MakeTable({"Time", "Sequence", "Frame", "Command", "Submit",
	                                     "Action", "Image", "Address", "Size", "Owner", "Dirty",
	                                     "Shader", "Note"});
	m_pixel_watches = MakeTable({"Watch", "Pixel", "Backing byte", "Resolution", "Time", "Frame",
	                             "Command", "Action", "Image", "Writer", "Owner", "Status"});
	m_gpu_break_conditions = MakeTable({"ID", "Kind", "Match", "Action", "State", "Hits",
	                                   "Last reason"});
	m_io_events = MakeTable({"Time", "Operation", "FD", "Guest path", "Host resource", "Offset",
	                         "Requested", "Result", "Thread", "Module", "Caller"});
	m_io_files = MakeTable({"Guest path", "Host resource", "Opens", "Closes", "Reads", "Writes",
	                       "Seeks", "Stats", "Bytes read", "Bytes written", "Last activity", "Module"});

	// CPU debugger: thread selection on the left, halted state on the right.
	auto* debugger_split = new QSplitter();
	debugger_split->addWidget(m_threads);
	auto* cpu_tabs = new QTabWidget();
	cpu_tabs->addTab(m_registers, "Registers");
	cpu_tabs->addTab(m_callstack, "Call stack");
	cpu_tabs->addTab(m_disassembly, "Disassembly");
	auto* breakpoint_page = new QWidget();
	auto* breakpoint_layout = new QVBoxLayout(breakpoint_page);
	auto* breakpoint_bar = new QHBoxLayout();
	m_breakpoint_location = new QLineEdit();
	m_breakpoint_location->setPlaceholderText("symbol, module+offset, or 0x address");
	auto* add_breakpoint = new QPushButton("Add breakpoint");
	auto* remove_breakpoint = new QPushButton("Remove selected");
	breakpoint_bar->addWidget(m_breakpoint_location, 1);
	breakpoint_bar->addWidget(add_breakpoint);
	breakpoint_bar->addWidget(remove_breakpoint);
	breakpoint_layout->addLayout(breakpoint_bar);
	breakpoint_layout->addWidget(m_breakpoints, 1);
	cpu_tabs->addTab(breakpoint_page, "Breakpoints");
	debugger_split->addWidget(cpu_tabs);
	debugger_split->setStretchFactor(0, 2);
	debugger_split->setStretchFactor(1, 3);
	tabs->addTab(debugger_split, "Debugger");

	// Bounded memory inspector. Writes stay explicit; browsing never mutates the guest.
	auto* memory_page = new QWidget();
	auto* memory_layout = new QVBoxLayout(memory_page);
	auto* memory_bar = new QHBoxLayout();
	m_memory_address = new QLineEdit("0x0");
	m_memory_size = new QLineEdit("256");
	auto* read_memory = new QPushButton("Read memory");
	m_memory = new QPlainTextEdit();
	m_memory->setReadOnly(true);
	m_memory->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
	memory_bar->addWidget(new QLabel("Address"));
	memory_bar->addWidget(m_memory_address, 1);
	memory_bar->addWidget(new QLabel("Bytes"));
	memory_bar->addWidget(m_memory_size);
	memory_bar->addWidget(read_memory);
	memory_layout->addLayout(memory_bar);
	memory_layout->addWidget(m_memory, 1);
	tabs->addTab(memory_page, "Memory");

	// Graphics: frame commands, shader code/resources, and image ownership history.
	m_graphics_tabs = new QTabWidget();
	auto* frame_page = new QWidget();
	auto* frame_layout = new QVBoxLayout(frame_page);
	auto* frame_bar = new QHBoxLayout();
	auto* capture_command = new QPushButton("Save command bundle");
	m_renderdoc_capture = new QPushButton("Capture next frame in RenderDoc");
	m_renderdoc_status = new QLabel("RenderDoc status unavailable until attached");
	m_renderdoc_status->setTextInteractionFlags(Qt::TextSelectableByMouse);
	m_command_capture_status = new QLabel(
	    "Select a draw or dispatch. The bundle preserves exact top-level PM4 and reports what is "
	    "still missing for deterministic execution.");
	m_command_capture_status->setTextInteractionFlags(Qt::TextSelectableByMouse);
	frame_bar->addWidget(m_command_capture_status, 1);
	frame_bar->addWidget(capture_command);
	frame_layout->addLayout(frame_bar);
	auto* renderdoc_bar = new QHBoxLayout();
	renderdoc_bar->addWidget(m_renderdoc_status, 1);
	renderdoc_bar->addWidget(m_renderdoc_capture);
	frame_layout->addLayout(renderdoc_bar);
	frame_layout->addWidget(m_frame, 1);
	m_graphics_tabs->addTab(frame_page, "Last frame");
	auto* shader_split = new QSplitter();
	shader_split->addWidget(m_shaders);
	auto* shader_detail = new QWidget();
	auto* shader_detail_layout = new QVBoxLayout(shader_detail);
	auto* shader_bar = new QHBoxLayout();
	m_shader_title = new QLabel("Select a shader");
	auto* dump_shader = new QPushButton("Save to _Shaders");
	shader_bar->addWidget(m_shader_title, 1);
	shader_bar->addWidget(dump_shader);
	shader_detail_layout->addLayout(shader_bar);
	auto* shader_tabs = new QTabWidget();
	m_shader_isa = new QPlainTextEdit();
	m_shader_ir = new QPlainTextEdit();
	m_shader_spirv = new QPlainTextEdit();
	for (auto* editor: {m_shader_isa, m_shader_ir, m_shader_spirv}) {
		editor->setReadOnly(true);
		editor->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
	}
	shader_tabs->addTab(m_shader_isa, "RDNA2");
	shader_tabs->addTab(m_shader_ir, "IR");
	shader_tabs->addTab(m_shader_spirv, "SPIR-V");
	auto* shader_resources = new QSplitter(Qt::Vertical);
	shader_resources->addWidget(m_resources);
	auto* shader_preview_page = new QWidget();
	auto* shader_preview_layout = new QVBoxLayout(shader_preview_page);
	auto* shader_preview_bar = new QHBoxLayout();
	m_shader_resource_preview_info =
	    new QLabel("Select an image resource to preview its current native texture");
	auto* refresh_shader_preview = new QPushButton("Refresh texture");
	auto* trace_shader_resource = new QPushButton("Trace resource");
	shader_preview_bar->addWidget(m_shader_resource_preview_info, 1);
	shader_preview_bar->addWidget(trace_shader_resource);
	shader_preview_bar->addWidget(refresh_shader_preview);
	m_shader_resource_preview = new QLabel("No texture selected");
	m_shader_resource_preview->setAlignment(Qt::AlignCenter);
	m_shader_resource_preview->setMinimumHeight(160);
	shader_preview_layout->addLayout(shader_preview_bar);
	shader_preview_layout->addWidget(m_shader_resource_preview, 1);
	shader_resources->addWidget(shader_preview_page);
	shader_resources->setStretchFactor(0, 3);
	shader_resources->setStretchFactor(1, 2);
	shader_tabs->addTab(shader_resources, "Resources");
	shader_detail_layout->addWidget(shader_tabs, 1);
	shader_split->addWidget(shader_detail);
	shader_split->setStretchFactor(0, 2);
	shader_split->setStretchFactor(1, 3);
	m_graphics_tabs->addTab(shader_split, "Shaders");
	auto* resource_split = new QSplitter();
	resource_split->addWidget(m_resource_history);
	auto* preview_page = new QWidget();
	auto* preview_layout = new QVBoxLayout(preview_page);
	auto* preview_bar = new QHBoxLayout();
	m_resource_preview_info = new QLabel("Select an image event to request a preview");
	auto* refresh_preview = new QPushButton("Refresh preview");
	auto* trace_history_resource = new QPushButton("Trace range");
	preview_bar->addWidget(m_resource_preview_info, 1);
	preview_bar->addWidget(trace_history_resource);
	preview_bar->addWidget(refresh_preview);
	m_resource_preview = new QLabel("No preview");
	m_resource_preview->setAlignment(Qt::AlignCenter);
	m_resource_preview->setMinimumSize(320, 180);
	m_resource_preview->setScaledContents(false);
	preview_layout->addLayout(preview_bar);
	preview_layout->addWidget(m_resource_preview, 1);
	resource_split->addWidget(preview_page);
	resource_split->setStretchFactor(0, 3);
	resource_split->setStretchFactor(1, 2);
	m_graphics_tabs->addTab(resource_split, "Resource history");

	// Resource provenance combines current aliases with a bounded chronological history. It is
	// intentionally metadata-only until the user selects one row for an explicit GPU preview.
	m_resource_trace_page = new QWidget();
	auto* trace_layout = new QVBoxLayout(m_resource_trace_page);
	auto* trace_bar = new QHBoxLayout();
	m_resource_trace_address = new QLineEdit("0x0");
	m_resource_trace_size = new QLineEdit("0x1");
	auto* refresh_trace = new QPushButton("Trace range");
	m_resource_trace_summary = new QLabel("Enter a guest address or trace a selected resource");
	m_resource_trace_summary->setTextInteractionFlags(Qt::TextSelectableByMouse);
	trace_bar->addWidget(new QLabel("Guest address"));
	trace_bar->addWidget(m_resource_trace_address, 1);
	trace_bar->addWidget(new QLabel("Bytes"));
	trace_bar->addWidget(m_resource_trace_size);
	trace_bar->addWidget(refresh_trace);
	trace_layout->addLayout(trace_bar);
	trace_layout->addWidget(m_resource_trace_summary);
	auto* pixel_bar = new QHBoxLayout();
	m_pixel_watch_x = new QLineEdit("0");
	m_pixel_watch_y = new QLineEdit("0");
	m_pixel_watch_x->setMaximumWidth(90);
	m_pixel_watch_y->setMaximumWidth(90);
	auto* add_pixel_watch = new QPushButton("Watch pixel");
	auto* remove_pixel_watch = new QPushButton("Remove selected watch");
	auto* clear_pixel_watches = new QPushButton("Clear watches");
	pixel_bar->addWidget(new QLabel("Pixel X"));
	pixel_bar->addWidget(m_pixel_watch_x);
	pixel_bar->addWidget(new QLabel("Y"));
	pixel_bar->addWidget(m_pixel_watch_y);
	pixel_bar->addWidget(add_pixel_watch);
	pixel_bar->addWidget(remove_pixel_watch);
	pixel_bar->addWidget(clear_pixel_watches);
	pixel_bar->addStretch(1);
	trace_layout->addLayout(pixel_bar);
	m_resource_alias_map = new ResourceAliasMapWidget();
	trace_layout->addWidget(m_resource_alias_map);
	auto* trace_split = new QSplitter();
	auto* trace_tables = new QSplitter(Qt::Vertical);
	trace_tables->addWidget(m_resource_trace_aliases);
	auto* trace_event_tabs = new QTabWidget();
	trace_event_tabs->addTab(m_resource_trace_events, "Resource events");
	trace_event_tabs->addTab(m_pixel_watches, "Pixel watchpoints");
	auto* gpu_break_page = new QWidget();
	auto* gpu_break_layout = new QVBoxLayout(gpu_break_page);
	auto* gpu_break_bar = new QHBoxLayout();
	m_gpu_break_kind = new QComboBox();
	m_gpu_break_kind->addItem("Shader hash / base", "shader");
	m_gpu_break_kind->addItem("Resource address", "resource");
	m_gpu_break_kind->addItem("Pixel watchpoint", "pixel");
	m_gpu_break_kind->addItem("NaN / Inf preview", "nonfinite");
	m_gpu_break_value = new QLineEdit("0x0");
	m_gpu_break_value->setPlaceholderText("hash, address, watch ID; 0 = any NaN/Inf");
	m_gpu_break_action = new QLineEdit();
	m_gpu_break_action->setPlaceholderText("optional resource action contains");
	m_gpu_break_one_shot = new QCheckBox("one shot");
	auto* add_gpu_break = new QPushButton("Add condition");
	auto* remove_gpu_break = new QPushButton("Remove selected");
	auto* clear_gpu_breaks = new QPushButton("Clear");
	gpu_break_bar->addWidget(m_gpu_break_kind);
	gpu_break_bar->addWidget(m_gpu_break_value, 1);
	gpu_break_bar->addWidget(m_gpu_break_action, 1);
	gpu_break_bar->addWidget(m_gpu_break_one_shot);
	gpu_break_bar->addWidget(add_gpu_break);
	gpu_break_bar->addWidget(remove_gpu_break);
	gpu_break_bar->addWidget(clear_gpu_breaks);
	gpu_break_layout->addLayout(gpu_break_bar);
	gpu_break_layout->addWidget(m_gpu_break_conditions, 1);
	trace_event_tabs->addTab(gpu_break_page, "Break conditions");
	trace_tables->addWidget(trace_event_tabs);
	trace_tables->setStretchFactor(0, 2);
	trace_tables->setStretchFactor(1, 3);
	trace_split->addWidget(trace_tables);
	auto* trace_preview_page = new QWidget();
	auto* trace_preview_layout = new QVBoxLayout(trace_preview_page);
	auto* trace_preview_bar = new QHBoxLayout();
	m_resource_trace_preview_info = new QLabel("Select an alias or event to preview its current image");
	auto* refresh_trace_preview = new QPushButton("Refresh preview");
	trace_preview_bar->addWidget(m_resource_trace_preview_info, 1);
	trace_preview_bar->addWidget(refresh_trace_preview);
	m_resource_trace_preview = new QLabel("No preview");
	m_resource_trace_preview->setAlignment(Qt::AlignCenter);
	m_resource_trace_preview->setMinimumSize(320, 180);
	trace_preview_layout->addLayout(trace_preview_bar);
	trace_preview_layout->addWidget(m_resource_trace_preview, 1);
	trace_split->addWidget(trace_preview_page);
	trace_split->setStretchFactor(0, 4);
	trace_split->setStretchFactor(1, 2);
	trace_layout->addWidget(trace_split, 1);
	m_graphics_tabs->addTab(m_resource_trace_page, "Resource trace");
	tabs->addTab(m_graphics_tabs, "Graphics");

	// Guest filesystem activity, kept separate from the CPU debugger so high-volume file traces
	// remain searchable without displacing stopped-thread state.
	auto* io_page = new QWidget();
	auto* io_layout = new QVBoxLayout(io_page);
	auto* io_bar = new QHBoxLayout();
	m_io_filter = new QLineEdit();
	m_io_filter->setPlaceholderText("Filter operation, guest path, host resource, or module");
	auto* refresh_io = new QPushButton("Refresh I/O");
	io_bar->addWidget(m_io_filter, 1);
	io_bar->addWidget(refresh_io);
	io_layout->addLayout(io_bar);
	auto* io_tabs = new QTabWidget();
	io_tabs->addTab(m_io_events, "Events");
	io_tabs->addTab(m_io_files, "Files (session)");
	io_layout->addWidget(io_tabs, 1);
	tabs->addTab(io_page, "I/O");

	// Module inventory and a searchable symbol index, matching the overlay's two views.
	auto* lookup_tabs = new QTabWidget();
	lookup_tabs->addTab(m_modules, "Modules");
	auto* symbols_page = new QWidget();
	auto* symbols_layout = new QVBoxLayout(symbols_page);
	auto* symbols_bar = new QHBoxLayout();
	m_symbol_filter = new QLineEdit();
	m_symbol_filter->setPlaceholderText("Filter symbols");
	m_symbol_module = new QComboBox();
	m_symbol_module->addItem("All modules", QString());
	symbols_bar->addWidget(m_symbol_filter, 1);
	symbols_bar->addWidget(m_symbol_module);
	symbols_layout->addLayout(symbols_bar);
	symbols_layout->addWidget(m_symbols, 1);
	lookup_tabs->addTab(symbols_page, "Symbols");
	tabs->addTab(lookup_tabs, "Modules & symbols");
	layout->addWidget(tabs, 1);
	setCentralWidget(central);

	m_socket = new QLocalSocket(this);
	m_timer = new QTimer(this);
	m_timer->setInterval(750);
	m_discovery = new QTimer(this);
	m_discovery->setInterval(1000);

	connect(refresh, &QPushButton::clicked, this, [this] { RefreshSessions(); });
	connect(m_connect, &QPushButton::clicked, this, [this] {
		if (m_socket->state() == QLocalSocket::ConnectedState) Disconnect();
		else ConnectSelected();
	});
	connect(m_pause, &QPushButton::clicked, this, [this] { FillSummary(Request("pause")); });
	connect(m_continue, &QPushButton::clicked, this, [this] { FillSummary(Request("continue")); });
	const auto step = [this](const QString& mode) {
		if (m_selected_thread != 0)
			FillSummary(Request("step", {{"thread", m_selected_thread}, {"mode", mode}}));
	};
	connect(m_step_into, &QPushButton::clicked, this, [step] { step("into"); });
	connect(m_step_over, &QPushButton::clicked, this, [step] { step("over"); });
	connect(m_step_out, &QPushButton::clicked, this, [step] { step("out"); });
	connect(m_threads, &QTableWidget::currentCellChanged, this,
	        [this](int row, int, int, int) {
		        if (row < 0 || m_threads->item(row, 0) == nullptr) return;
		        m_selected_thread = m_threads->item(row, 0)->text().toInt();
		        FillStoppedDetails(Request("stopped_details", {{"thread", m_selected_thread}}));
	        });
	connect(m_shaders, &QTableWidget::currentCellChanged, this,
	        [this](int row, int, int, int) {
		        if (row < 0 || m_shaders->item(row, 2) == nullptr) return;
		        LoadShader(m_shaders->item(row, 2)->data(Qt::UserRole).toULongLong());
	        });
	connect(m_resources, &QTableWidget::currentCellChanged, this,
	        [this](int row, int, int, int) {
		        if (row < 0 || m_resources->item(row, 0) == nullptr ||
		            m_resources->item(row, 3) == nullptr) return;
		        if (m_resources->item(row, 0)->text() != "image") {
			        m_shader_resource_preview_info->setText(
			            "This resource is not an image; its decoded descriptor is shown above");
			        return;
		        }
		        const auto address = m_resources->item(row, 3)->data(Qt::UserRole).toULongLong();
		        if (address == 0) {
			        m_shader_resource_preview_info->setText("The selected image has a null address");
			        return;
		        }
		        m_preview_target = PreviewTarget::Shader;
		        m_preview_address = address;
		        // Shader descriptors do not carry the complete backing allocation size. A zero size
		        // asks the texture cache for the exact live image whose base matches this address.
		        m_preview_size = 0;
		        RefreshResourcePreview(true);
	        });
	connect(m_resource_history, &QTableWidget::currentCellChanged, this,
	        [this](int row, int, int, int) {
		        if (row < 0 || m_resource_history->item(row, 3) == nullptr ||
		            m_resource_history->item(row, 4) == nullptr) return;
		        m_preview_target = PreviewTarget::ResourceHistory;
		        m_preview_address = m_resource_history->item(row, 3)->data(Qt::UserRole).toULongLong();
		        m_preview_size = m_resource_history->item(row, 4)->data(Qt::UserRole).toULongLong();
		        RefreshResourcePreview(true);
	        });
	connect(refresh_preview, &QPushButton::clicked, this,
	        [this] {
		        m_preview_target = PreviewTarget::ResourceHistory;
		        RefreshResourcePreview(true);
	        });
	connect(refresh_shader_preview, &QPushButton::clicked, this,
	        [this] {
		        m_preview_target = PreviewTarget::Shader;
		        RefreshResourcePreview(true);
	        });
	connect(trace_shader_resource, &QPushButton::clicked, this,
	        [this] { TraceRange(m_preview_address, m_preview_size == 0 ? 1 : m_preview_size); });
	connect(trace_history_resource, &QPushButton::clicked, this,
	        [this] { TraceRange(m_preview_address, m_preview_size == 0 ? 1 : m_preview_size); });
	connect(refresh_trace, &QPushButton::clicked, this, [this] { RefreshResourceTrace(); });
	connect(add_pixel_watch, &QPushButton::clicked, this, [this] {
		bool address_ok = false;
		bool x_ok = false;
		bool y_ok = false;
		const auto address = m_resource_trace_address->text().trimmed().toULongLong(&address_ok, 0);
		const auto x = m_pixel_watch_x->text().trimmed().toUInt(&x_ok, 0);
		const auto y = m_pixel_watch_y->text().trimmed().toUInt(&y_ok, 0);
		if (!address_ok || address == 0 || !x_ok || !y_ok) {
			m_resource_trace_summary->setText("Enter a valid resource base address and pixel X/Y");
			return;
		}
		FillPixelWatches(Request("pixel_watch_add", {{"address", QString::number(address)},
		                                                    {"x", static_cast<qint64>(x)},
		                                                    {"y", static_cast<qint64>(y)}}));
	});
	connect(remove_pixel_watch, &QPushButton::clicked, this, [this] {
		const auto row = m_pixel_watches->currentRow();
		if (row < 0 || m_pixel_watches->item(row, 0) == nullptr) return;
		const auto id = m_pixel_watches->item(row, 0)->data(Qt::UserRole).toULongLong();
		if (id != 0) FillPixelWatches(Request("pixel_watch_remove", {{"id", QString::number(id)}}));
	});
	connect(clear_pixel_watches, &QPushButton::clicked, this,
	        [this] { FillPixelWatches(Request("pixel_watch_clear")); });
	connect(add_gpu_break, &QPushButton::clicked, this, [this] {
		bool value_ok = false;
		const auto value = m_gpu_break_value->text().trimmed().toULongLong(&value_ok, 0);
		if (!value_ok) {
			m_resource_trace_summary->setText("Enter a valid GPU break-condition match value");
			return;
		}
		FillGpuBreakConditions(Request("gpu_break_condition_add",
		                               {{"kind", m_gpu_break_kind->currentData().toString()},
		                                {"value", QString::number(value)},
		                                {"action", m_gpu_break_action->text().trimmed()},
		                                {"one_shot", m_gpu_break_one_shot->isChecked()}}));
	});
	connect(remove_gpu_break, &QPushButton::clicked, this, [this] {
		const auto row = m_gpu_break_conditions->currentRow();
		if (row < 0 || m_gpu_break_conditions->item(row, 0) == nullptr) return;
		const auto id = m_gpu_break_conditions->item(row, 0)->data(Qt::UserRole).toULongLong();
		if (id != 0)
			FillGpuBreakConditions(Request("gpu_break_condition_remove",
			                               {{"id", QString::number(id)}}));
	});
	connect(clear_gpu_breaks, &QPushButton::clicked, this,
	        [this] { FillGpuBreakConditions(Request("gpu_break_condition_clear")); });
	connect(m_resource_trace_address, &QLineEdit::returnPressed, this,
	        [this] { RefreshResourceTrace(); });
	connect(m_resource_trace_size, &QLineEdit::returnPressed, this,
	        [this] { RefreshResourceTrace(); });
	connect(refresh_trace_preview, &QPushButton::clicked, this,
	        [this] {
		        m_preview_target = PreviewTarget::ResourceTrace;
		        RefreshResourcePreview(true);
	        });
	const auto select_trace_preview = [this](QTableWidget* table, int row) {
		if (row < 0 || table->item(row, 0) == nullptr) return;
		const auto address = table->item(row, 0)->data(Qt::UserRole).toULongLong();
		const auto size = table->item(row, 0)->data(Qt::UserRole + 1).toULongLong();
		if (address == 0) return;
		m_preview_target = PreviewTarget::ResourceTrace;
		m_preview_address = address;
		m_preview_size = size;
		RefreshResourcePreview(true);
	};
	connect(m_resource_trace_aliases, &QTableWidget::currentCellChanged, this,
	        [select_trace_preview, this](int row, int, int, int) {
		        select_trace_preview(m_resource_trace_aliases, row);
	        });
	connect(m_resource_trace_events, &QTableWidget::currentCellChanged, this,
	        [select_trace_preview, this](int row, int, int, int) {
		        select_trace_preview(m_resource_trace_events, row);
	        });
	connect(refresh_io, &QPushButton::clicked, this, [this] { RefreshIo(); });
	connect(m_io_filter, &QLineEdit::returnPressed, this, [this] { RefreshIo(); });
	connect(dump_shader, &QPushButton::clicked, this, [this] {
		if (m_selected_shader == 0) return;
		const auto result = Request("dump_shader", {{"hash", QString::number(m_selected_shader)}});
		if (result.value("ok").toBool())
			m_shader_title->setText(Hex(m_selected_shader) + " — saved to " + result.value("path").toString());
	});
	connect(capture_command, &QPushButton::clicked, this, [this] {
		const auto row = m_frame->currentRow();
		if (row < 0 || m_frame->item(row, 2) == nullptr) {
			m_command_capture_status->setText("Select a draw or dispatch first");
			return;
		}
		const auto submit = m_frame->item(row, 2)->data(Qt::UserRole).toULongLong();
		const auto result = Request("capture_command", {{"submit", QString::number(submit)}}, 5000);
		if (!result.value("ok").toBool()) {
			m_command_capture_status->setText(result.value("error").toString());
			return;
		}
		m_command_capture_status->setText(
		    QString("Saved submit %1 (%2 + %3 words) to %4 — %5")
		        .arg(submit)
		        .arg(JsonU64(result.value("command_words")))
		        .arg(JsonU64(result.value("constant_words")))
		        .arg(result.value("path").toString())
		        .arg(result.value("status").toString()));
	});
	connect(m_renderdoc_capture, &QPushButton::clicked, this, [this] {
		FillRenderDocStatus(Request("renderdoc_capture"));
	});
	connect(m_symbol_filter, &QLineEdit::textChanged, this, [this] { RefreshSymbols(); });
	connect(m_symbol_module, &QComboBox::currentIndexChanged, this, [this] { RefreshSymbols(); });
	connect(add_breakpoint, &QPushButton::clicked, this, [this] {
		if (!m_breakpoint_location->text().trimmed().isEmpty()) {
			Request("breakpoint_add", {{"location", m_breakpoint_location->text().trimmed()}});
			FillBreakpoints(Request("breakpoints"));
		}
	});
	connect(remove_breakpoint, &QPushButton::clicked, this, [this] {
		const auto row = m_breakpoints->currentRow();
		if (row >= 0 && m_breakpoints->item(row, 0) != nullptr) {
			FillBreakpoints(Request("breakpoint_remove", {{"id", m_breakpoints->item(row, 0)->text().toInt()}}));
		}
	});
	connect(read_memory, &QPushButton::clicked, this, [this] { ReadMemory(); });
	connect(m_timer, &QTimer::timeout, this, [this] { RefreshData(); });
	connect(m_discovery, &QTimer::timeout, this, [this] {
		if (m_socket->state() == QLocalSocket::ConnectedState) return;
		RefreshSessions();
		if (m_descriptors.size() == 1) ConnectSelected();
	});
	connect(m_socket, &QLocalSocket::disconnected, this,
	        [this] { SetConnectionState("Disconnected", false); });

	SetConnectionState("Not attached", false);
	RefreshSessions();
	m_discovery->start();
}

MainWindow::~MainWindow() = default;

void MainWindow::RefreshSessions() {
	m_descriptors.clear();
	m_sessions->clear();
	const auto local = qEnvironmentVariable("LOCALAPPDATA");
	const QDir folder(QDir(local).filePath("KytyPS5/DebuggerSessions"));
	for (const auto& name: folder.entryList({"*.json"}, QDir::Files, QDir::Time)) {
		QFile file(folder.filePath(name));
		if (!file.open(QIODevice::ReadOnly)) continue;
		const auto document = QJsonDocument::fromJson(file.readAll());
		const auto object = document.object();
		if (object.value("schema") != "kyty_debugger_session" || object.value("protocol") != 1)
			continue;
		SessionDescriptor descriptor;
		descriptor.pid = JsonU64(object.value("pid"));
		descriptor.endpoint = object.value("endpoint").toString();
		descriptor.token = object.value("token").toString();
		if (descriptor.endpoint.isEmpty() || descriptor.token.isEmpty() ||
		    !ProcessIsRunning(descriptor.pid)) continue;
		m_descriptors.push_back(descriptor);
		m_sessions->addItem(QString("Kyty process %1 — %2").arg(descriptor.pid).arg(descriptor.endpoint));
	}
	if (m_descriptors.isEmpty()) m_sessions->addItem("No attachable Kyty sessions found");
}

void MainWindow::ConnectSelected() {
	const auto index = m_sessions->currentIndex();
	if (index < 0 || index >= m_descriptors.size()) return;
	m_socket->abort();
	m_token = m_descriptors[index].token;
	m_socket->connectToServer(m_descriptors[index].endpoint, QIODevice::ReadWrite);
	if (!m_socket->waitForConnected(1500)) {
		SetConnectionState("Attach failed: " + m_socket->errorString(), false);
		return;
	}
	SetConnectionState(QString("Attached to process %1").arg(m_descriptors[index].pid), true);
	RefreshData();
	m_timer->start();
}

void MainWindow::Disconnect() {
	m_timer->stop();
	m_socket->disconnectFromServer();
	m_token.clear();
	SetConnectionState("Not attached", false);
}

QJsonObject MainWindow::Request(const QString& command, const QJsonObject& arguments,
                                int timeout_ms) {
	if (m_socket->state() != QLocalSocket::ConnectedState) return {};
	auto object = arguments;
	object.insert("command", command);
	object.insert("token", m_token);
	const auto request = QJsonDocument(object)
	                         .toJson(QJsonDocument::Compact) + '\n';
	if (m_socket->write(request) != request.size() || !m_socket->waitForBytesWritten(timeout_ms))
		return {};
	while (!m_socket->canReadLine()) {
		if (!m_socket->waitForReadyRead(timeout_ms)) return {};
	}
	return QJsonDocument::fromJson(m_socket->readLine()).object();
}

void MainWindow::RefreshData() {
	if (m_socket->state() != QLocalSocket::ConnectedState) return;
	FillSummary(Request("summary"));
	FillThreads(Request("threads"));
	FillModules(Request("modules"));
	FillShaders(Request("shaders"));
	FillFrame(Request("frame"));
	FillRenderDocStatus(Request("renderdoc_status"));
	FillBreakpoints(Request("breakpoints"));
	FillResourceHistory(Request("resource_history", {{"limit", 512}}));
	RefreshIo();
	if (m_graphics_tabs->currentWidget() == m_resource_trace_page) {
		RefreshResourceTrace();
		RefreshPixelWatches();
		RefreshGpuBreakConditions();
	}
	if (m_preview_address != 0) RefreshResourcePreview(false);
	if (m_selected_thread != 0)
		FillStoppedDetails(Request("stopped_details", {{"thread", m_selected_thread}}));
}

void MainWindow::FillSummary(const QJsonObject& response) {
	if (!response.value("ok").toBool()) return;
	const auto graphics = response.value("graphics").toObject();
	m_summary->setText(QString("%1  |  Frame %2  |  Last frame: %3 draws, %4 dispatches  |  %5 shaders  |  Totals: %6 draws, %7 dispatches")
	                       .arg(response.value("paused").toBool() ? "Paused" : "Running")
	                       .arg(JsonU64(graphics.value("frame")))
	                       .arg(JsonU64(graphics.value("draws_last_frame")))
	                       .arg(JsonU64(graphics.value("dispatches_last_frame")))
	                       .arg(JsonU64(graphics.value("shader_count")))
	                       .arg(JsonU64(graphics.value("total_draws")))
	                       .arg(JsonU64(graphics.value("total_dispatches"))));
}

void MainWindow::FillRenderDocStatus(const QJsonObject& response) {
	if (!response.value("ok").toBool()) return;
	const auto available = response.value("available").toBool();
	const auto state = response.value("state").toString();
	const auto accepted = response.value("accepted");
	if (!available) {
		m_renderdoc_status->setText(
		    "RenderDoc is unavailable — enable the launcher's RenderDoc option before starting the game");
		m_renderdoc_capture->setEnabled(false);
		return;
	}
	m_renderdoc_capture->setEnabled(state == "idle");
	QString text = "RenderDoc: " + state;
	if (!accepted.isUndefined() && !accepted.toBool()) text += " (request was already pending)";
	const auto path = response.value("capture_path").toString();
	if (!path.isEmpty()) text += "  |  template " + path;
	if (response.value("has_result").toBool()) {
		text += response.value("last_succeeded").toBool() ? "  |  last capture succeeded"
		                                                : "  |  last capture failed";
	}
	text += QString("  |  completed %1").arg(JsonU64(response.value("completed_captures")));
	m_renderdoc_status->setText(text);
}

void MainWindow::FillThreads(const QJsonObject& response) {
	TableUpdateGuard update(m_threads);
	const auto values = response.value("threads").toArray();
	m_threads->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_threads, row, 0, QString::number(value.value("id").toInt()));
		SetItem(m_threads, row, 1, QString::number(value.value("guest_id").toInt()));
		SetItem(m_threads, row, 2, QString::number(JsonU64(value.value("host_id"))));
		SetItem(m_threads, row, 3, value.value("name").toString());
		SetItem(m_threads, row, 4, value.value("stopped").toBool() ? value.value("reason").toString() : "running");
		SetItem(m_threads, row, 5, value.contains("address") ? Hex(JsonU64(value.value("address"))) : QString());
	}
}

void MainWindow::FillModules(const QJsonObject& response) {
	TableUpdateGuard update(m_modules);
	const auto values = response.value("modules").toArray();
	const auto selected_module = m_symbol_module->currentData().toString();
	m_symbol_module->blockSignals(true);
	m_symbol_module->clear();
	m_symbol_module->addItem("All modules", QString());
	m_modules->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		const auto name = value.value("name").toString();
		SetItem(m_modules, row, 0, QString::number(value.value("id").toInt()));
		SetItem(m_modules, row, 1, name);
		SetItem(m_modules, row, 2, Hex(JsonU64(value.value("base"))));
		SetItem(m_modules, row, 3, Hex(JsonU64(value.value("size"))));
		m_symbol_module->addItem(name, name);
	}
	const auto selected_index = m_symbol_module->findData(selected_module);
	m_symbol_module->setCurrentIndex(selected_index >= 0 ? selected_index : 0);
	m_symbol_module->blockSignals(false);
	if (m_symbols->rowCount() == 0 && !values.isEmpty()) RefreshSymbols();
}

void MainWindow::FillShaders(const QJsonObject& response) {
	TableUpdateGuard update(m_shaders);
	const auto values = response.value("shaders").toArray();
	m_shaders->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_shaders, row, 0, QString::number(value.value("sequence").toInt()));
		SetItem(m_shaders, row, 1, value.value("stage").toString());
		const auto hash = JsonU64(value.value("hash"));
		SetItem(m_shaders, row, 2, Hex(hash));
		m_shaders->item(row, 2)->setData(Qt::UserRole, QVariant::fromValue(hash));
		SetItem(m_shaders, row, 3, Hex(JsonU64(value.value("base"))));
		SetItem(m_shaders, row, 4, QString::number(value.value("gcn_bytes").toInt()));
		SetItem(m_shaders, row, 5, QString::number(value.value("spirv_words").toInt()));
		SetItem(m_shaders, row, 6, QString::number(value.value("resource_count").toInt()));
	}
}

void MainWindow::FillStoppedDetails(const QJsonObject& response) {
	if (!response.value("ok").toBool()) return;
	TableUpdateGuard registers_update(m_registers);
	TableUpdateGuard callstack_update(m_callstack);
	TableUpdateGuard disassembly_update(m_disassembly);
	const auto registers = response.value("registers").toObject();
	static const QStringList names = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
	                                  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	                                  "rip", "rflags"};
	m_registers->setRowCount(names.size());
	for (int row = 0; row < names.size(); row++) {
		SetItem(m_registers, row, 0, names[row].toUpper());
		SetItem(m_registers, row, 1, Hex(JsonU64(registers.value(names[row]))));
	}

	const auto frames = response.value("frames").toArray();
	m_callstack->setRowCount(frames.size());
	for (int row = 0; row < frames.size(); row++) {
		const auto frame = frames[row].toObject();
		SetItem(m_callstack, row, 0, QString::number(row));
		SetItem(m_callstack, row, 1, Hex(JsonU64(frame.value("address"))));
		SetItem(m_callstack, row, 2, frame.value("description").toString());
	}

	const auto instructions = response.value("instructions").toArray();
	m_disassembly->setRowCount(instructions.size());
	for (int row = 0; row < instructions.size(); row++) {
		const auto instruction = instructions[row].toObject();
		SetItem(m_disassembly, row, 0, Hex(JsonU64(instruction.value("address"))));
		SetItem(m_disassembly, row, 1, instruction.value("bytes").toString());
		const auto symbol = instruction.value("symbol").toString();
		SetItem(m_disassembly, row, 2,
		        symbol.isEmpty() ? instruction.value("text").toString()
		                         : symbol + ":  " + instruction.value("text").toString());
	}
}

void MainWindow::FillBreakpoints(const QJsonObject& response) {
	TableUpdateGuard update(m_breakpoints);
	const auto values = response.value("breakpoints").toArray();
	m_breakpoints->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_breakpoints, row, 0, QString::number(value.value("id").toInt()));
		const auto label = value.value("label").toString();
		SetItem(m_breakpoints, row, 1,
		        label.isEmpty() ? Hex(JsonU64(value.value("address"))) : label);
		SetItem(m_breakpoints, row, 2,
		        value.value("pending").toBool() ? "pending"
		        : value.value("armed").toBool() ? "armed"
		                                         : "disabled");
		SetItem(m_breakpoints, row, 3, QString::number(JsonU64(value.value("hits"))));
	}
}

void MainWindow::RefreshSymbols() {
	if (m_socket->state() != QLocalSocket::ConnectedState) return;
	FillSymbols(Request("symbols", {{"filter", m_symbol_filter->text()},
	                                {"module", m_symbol_module->currentData().toString()},
	                                {"limit", 1000}}));
}

void MainWindow::FillSymbols(const QJsonObject& response) {
	TableUpdateGuard update(m_symbols);
	const auto values = response.value("symbols").toArray();
	m_symbols->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_symbols, row, 0, value.value("name").toString());
		SetItem(m_symbols, row, 1, value.value("module").toString());
		SetItem(m_symbols, row, 2, Hex(JsonU64(value.value("address"))));
	}
}

void MainWindow::LoadShader(quint64 hash) {
	if (hash == 0 || hash == m_selected_shader) return;
	m_selected_shader = hash;
	m_shader_title->setText(Hex(hash));
	FillShaderDetails(Request("shader_code", {{"hash", QString::number(hash)}}, 5000));
}

void MainWindow::FillShaderDetails(const QJsonObject& response) {
	if (!response.value("ok").toBool()) return;
	TableUpdateGuard update(m_resources);
	m_shader_isa->setPlainText(response.value("isa").toString());
	m_shader_ir->setPlainText(response.value("ir").toString());
	m_shader_spirv->setPlainText(response.value("spirv").toString());
	const auto values = response.value("resources").toArray();
	m_resources->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		const auto kind = value.value("kind").toString();
		QString access;
		if (value.value("read").toBool()) access += "R";
		if (value.value("written").toBool()) access += "W";
		if (value.value("atomic").toBool()) access += "A";
		QStringList descriptor;
		for (const auto dword: value.value("descriptor").toArray())
			descriptor << QString("%1").arg(dword.toVariant().toUInt(), 8, 16, QLatin1Char('0'));
		SetItem(m_resources, row, 0, kind);
		SetItem(m_resources, row, 1, QString::number(value.value("index").toInt()));
		SetItem(m_resources, row, 2, access);
		const auto address = JsonU64(value.value("address"));
		SetItem(m_resources, row, 3, address == 0 ? QString() : Hex(address));
		m_resources->item(row, 3)->setData(Qt::UserRole, QVariant::fromValue(address));
		const auto width = value.value("width").toInt();
		const auto height = value.value("height").toInt();
		const auto depth = value.value("depth").toInt();
		QString extent;
		if (width > 0) extent = QString("%1×%2×%3").arg(width).arg(height).arg(depth);
		else if (JsonU64(value.value("size")) > 0)
			extent = QString("%1 bytes").arg(JsonU64(value.value("size")));
		SetItem(m_resources, row, 4, extent);
		SetItem(m_resources, row, 5,
		        kind == "image"
		            ? QString("format %1 / tile %2").arg(value.value("format").toInt())
		                  .arg(value.value("tile").toInt())
		            : value.value("format").toInt() != 0
		                  ? QString("format %1").arg(value.value("format").toInt())
		                  : QString());
		SetItem(m_resources, row, 6, Hex(JsonU64(value.value("source"))));
		SetItem(m_resources, row, 7, Hex(JsonU64(value.value("first_use_pc"))));
		SetItem(m_resources, row, 8, descriptor.join(' '));
	}
	m_shader_resource_preview_info->setText(
	    values.isEmpty() ? "This shader has no captured resources"
	                     : "Select an image resource to preview its current native texture");
}

void MainWindow::FillResourceHistory(const QJsonObject& response) {
	TableUpdateGuard update(m_resource_history);
	const auto values = response.value("events").toArray();
	m_resource_history->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_resource_history, row, 0, QString::number(JsonU64(value.value("sequence"))));
		SetItem(m_resource_history, row, 1, QString::number(value.value("frame").toInt()));
		SetItem(m_resource_history, row, 2, value.value("action").toString());
		SetItem(m_resource_history, row, 3, Hex(JsonU64(value.value("address"))));
		SetItem(m_resource_history, row, 4, Hex(JsonU64(value.value("size"))));
		m_resource_history->item(row, 3)->setData(Qt::UserRole,
		                                             QVariant::fromValue(JsonU64(value.value("address"))));
		m_resource_history->item(row, 4)->setData(Qt::UserRole,
		                                             QVariant::fromValue(JsonU64(value.value("size"))));
		SetItem(m_resource_history, row, 5,
		        QString("%1×%2×%3").arg(value.value("width").toInt())
		            .arg(value.value("height").toInt()).arg(value.value("depth").toInt()));
		SetItem(m_resource_history, row, 6, QString::number(value.value("bpb").toInt()));
		SetItem(m_resource_history, row, 7,
		        QString("guest %1 / host %2").arg(value.value("guest_format").toInt())
		            .arg(value.value("host_format").toInt()));
		SetItem(m_resource_history, row, 8, QString::number(value.value("tile").toInt()));
	}
}

void MainWindow::TraceRange(quint64 address, quint64 size) {
	if (address == 0) return;
	m_resource_trace_address->setText(Hex(address));
	m_resource_trace_size->setText(Hex(size == 0 ? 1 : size));
	m_graphics_tabs->setCurrentWidget(m_resource_trace_page);
	RefreshResourceTrace();
}

void MainWindow::RefreshResourceTrace() {
	if (m_socket->state() != QLocalSocket::ConnectedState) return;
	bool address_ok = false;
	bool size_ok = false;
	const auto address = m_resource_trace_address->text().trimmed().toULongLong(&address_ok, 0);
	auto size = m_resource_trace_size->text().trimmed().toULongLong(&size_ok, 0);
	if (!address_ok || address == 0 || !size_ok) {
		m_resource_trace_summary->setText("Enter a valid non-zero guest address and byte size");
		return;
	}
	if (size == 0) size = 1;
	FillResourceTrace(Request("resource_trace", {{"address", QString::number(address)},
	                                             {"size", QString::number(size)},
	                                             {"limit", 4096}}, 5000));
}

void MainWindow::FillResourceTrace(const QJsonObject& response) {
	if (!response.value("ok").toBool()) {
		m_resource_trace_summary->setText(response.value("error").toString());
		return;
	}
	TableUpdateGuard aliases_update(m_resource_trace_aliases);
	TableUpdateGuard events_update(m_resource_trace_events);
	const auto alias_vertical = m_resource_trace_aliases->verticalScrollBar()->value();
	const auto alias_horizontal = m_resource_trace_aliases->horizontalScrollBar()->value();
	const auto event_vertical = m_resource_trace_events->verticalScrollBar()->value();
	const auto event_horizontal = m_resource_trace_events->horizontalScrollBar()->value();
	const auto aliases = response.value("aliases").toArray();
	const auto events = response.value("events").toArray();
	QVector<ResourceAliasMapWidget::Lane> alias_lanes;
	alias_lanes.reserve(aliases.size());
	m_resource_trace_aliases->setRowCount(aliases.size());
	for (int row = 0; row < aliases.size(); row++) {
		const auto value = aliases[row].toObject();
		const auto address = JsonU64(value.value("address"));
		const auto size = JsonU64(value.value("size"));
		alias_lanes.push_back({
		    ResourceIdentity(value) + (value.value("active").toBool() ? " active" : " retired"),
		    ResourceOwner(value), address, size, JsonU64(value.value("stencil_address")),
		    JsonU64(value.value("stencil_size")), JsonU64(value.value("metadata_address")),
		    JsonU64(value.value("metadata_size")), value.value("active").toBool()});
		SetItem(m_resource_trace_aliases, row, 0,
		        value.value("active").toBool()
		            ? (value.value("registered").toBool() ? "active" : "unregistered")
		            : "retired");
		m_resource_trace_aliases->item(row, 0)->setData(Qt::UserRole,
		                                                      QVariant::fromValue(address));
		m_resource_trace_aliases->item(row, 0)->setData(Qt::UserRole + 1,
		                                                      QVariant::fromValue(size));
		SetItem(m_resource_trace_aliases, row, 1, ResourceIdentity(value));
		SetItem(m_resource_trace_aliases, row, 2, Hex(address) + " + " + Hex(size));
		SetItem(m_resource_trace_aliases, row, 3, Hex(JsonU64(value.value("host_image"))));
		SetItem(m_resource_trace_aliases, row, 4, ResourceOwner(value));
		SetItem(m_resource_trace_aliases, row, 5, ResourceDirty(value));
		SetItem(m_resource_trace_aliases, row, 6, ResourceUsage(value));
		SetItem(m_resource_trace_aliases, row, 7,
		        QString("%1×%2×%3, pitch %4").arg(value.value("width").toInt())
		            .arg(value.value("height").toInt()).arg(value.value("depth").toInt())
		            .arg(value.value("pitch").toInt()));
		SetItem(m_resource_trace_aliases, row, 8,
		        QString("guest %1 / host %2 / tile %3 / %4 B")
		            .arg(value.value("guest_format").toInt()).arg(value.value("host_format").toInt())
		            .arg(value.value("tile").toInt()).arg(value.value("bpb").toInt()));
		SetItem(m_resource_trace_aliases, row, 9,
		        QString("%1 / %2 / %3").arg(value.value("samples").toInt())
		            .arg(value.value("levels").toInt()).arg(value.value("layers").toInt()));
		const auto stencil = JsonU64(value.value("stencil_address"));
		const auto metadata = JsonU64(value.value("metadata_address"));
		SetItem(m_resource_trace_aliases, row, 10,
		        stencil == 0 ? QString() : Hex(stencil) + " + " + Hex(JsonU64(value.value("stencil_size"))));
		SetItem(m_resource_trace_aliases, row, 11,
		        metadata == 0 ? QString()
		                      : QString("kind %1, %2 + %3").arg(value.value("metadata_kind").toInt())
		                            .arg(Hex(metadata)).arg(Hex(JsonU64(value.value("metadata_size")))));
		SetItem(m_resource_trace_aliases, row, 12,
		        value.value("action").toString() + " #" +
		            QString::number(JsonU64(value.value("sequence"))));
	}
	m_resource_alias_map->SetAliases(std::move(alias_lanes));
	m_resource_trace_events->setRowCount(events.size());
	for (int row = 0; row < events.size(); row++) {
		const auto value = events[row].toObject();
		const auto address = JsonU64(value.value("address"));
		const auto size = JsonU64(value.value("size"));
		SetItem(m_resource_trace_events, row, 0,
		        QString::number(JsonU64(value.value("timestamp_us")) / 1000.0, 'f', 3) + " ms");
		m_resource_trace_events->item(row, 0)->setData(Qt::UserRole,
		                                                     QVariant::fromValue(address));
		m_resource_trace_events->item(row, 0)->setData(Qt::UserRole + 1,
		                                                     QVariant::fromValue(size));
		SetItem(m_resource_trace_events, row, 1,
		        QString::number(JsonU64(value.value("sequence"))));
		SetItem(m_resource_trace_events, row, 2, QString::number(value.value("frame").toInt()));
		SetItem(m_resource_trace_events, row, 3,
		        value.value("has_command").toBool()
		            ? value.value("command_kind").toString() + " #" +
		                  QString::number(value.value("command_index").toInt())
		            : QString());
		SetItem(m_resource_trace_events, row, 4, QString::number(JsonU64(value.value("submit"))));
		SetItem(m_resource_trace_events, row, 5, value.value("action").toString());
		SetItem(m_resource_trace_events, row, 6, ResourceIdentity(value));
		SetItem(m_resource_trace_events, row, 7, Hex(address));
		SetItem(m_resource_trace_events, row, 8, Hex(size));
		SetItem(m_resource_trace_events, row, 9, ResourceOwner(value));
		SetItem(m_resource_trace_events, row, 10, ResourceDirty(value));
		SetItem(m_resource_trace_events, row, 11, ResourceShader(value));
		SetItem(m_resource_trace_events, row, 12, value.value("note").toString());
	}
	m_resource_trace_aliases->verticalScrollBar()->setValue(
	    (std::min)(alias_vertical, m_resource_trace_aliases->verticalScrollBar()->maximum()));
	m_resource_trace_aliases->horizontalScrollBar()->setValue(
	    (std::min)(alias_horizontal, m_resource_trace_aliases->horizontalScrollBar()->maximum()));
	m_resource_trace_events->verticalScrollBar()->setValue(
	    (std::min)(event_vertical, m_resource_trace_events->verticalScrollBar()->maximum()));
	m_resource_trace_events->horizontalScrollBar()->setValue(
	    (std::min)(event_horizontal, m_resource_trace_events->horizontalScrollBar()->maximum()));
	m_resource_trace_summary->setText(
	    QString("%1 aliases and %2 events overlap %3 + %4")
	        .arg(aliases.size()).arg(events.size()).arg(Hex(JsonU64(response.value("address"))))
	        .arg(Hex(JsonU64(response.value("size")))));
}

void MainWindow::RefreshPixelWatches() {
	if (m_socket->state() != QLocalSocket::ConnectedState) return;
	FillPixelWatches(Request("pixel_watches"));
}

void MainWindow::FillPixelWatches(const QJsonObject& response) {
	if (!response.value("ok").toBool()) {
		if (!response.value("error").toString().isEmpty())
			m_resource_trace_summary->setText(response.value("error").toString());
		return;
	}
	TableUpdateGuard update(m_pixel_watches);
	const auto vertical = m_pixel_watches->verticalScrollBar()->value();
	const auto horizontal = m_pixel_watches->horizontalScrollBar()->value();
	const auto watches = response.value("watches").toArray();
	const auto hits = response.value("hits").toArray();
	m_pixel_watches->setRowCount(hits.isEmpty() ? watches.size() : hits.size());
	if (hits.isEmpty()) {
		for (int row = 0; row < watches.size(); row++) {
			const auto watch = watches[row].toObject();
			const auto id = JsonU64(watch.value("id"));
			SetItem(m_pixel_watches, row, 0, QString::number(id));
			m_pixel_watches->item(row, 0)->setData(Qt::UserRole, QVariant::fromValue(id));
			SetItem(m_pixel_watches, row, 1,
			        QString("%1, %2").arg(watch.value("x").toInt()).arg(watch.value("y").toInt()));
			SetItem(m_pixel_watches, row, 2, Hex(JsonU64(watch.value("pixel_address"))));
			SetItem(m_pixel_watches, row, 3, watch.value("exact").toBool() ? "exact" : "pending");
			SetItem(m_pixel_watches, row, 11, watch.value("status").toString());
		}
	} else {
		for (int row = 0; row < hits.size(); row++) {
			const auto hit = hits[row].toObject();
			const auto event = hit.value("event").toObject();
			const auto id = JsonU64(hit.value("watch_id"));
			SetItem(m_pixel_watches, row, 0, QString::number(id));
			m_pixel_watches->item(row, 0)->setData(Qt::UserRole, QVariant::fromValue(id));
			SetItem(m_pixel_watches, row, 1,
			        QString("%1, %2").arg(hit.value("x").toInt()).arg(hit.value("y").toInt()));
			const auto byte_address = JsonU64(hit.value("pixel_address"));
			SetItem(m_pixel_watches, row, 2,
			        byte_address == 0 ? QString() : Hex(byte_address) + " + " +
			                                      QString::number(hit.value("pixel_size").toInt()));
			SetItem(m_pixel_watches, row, 3, hit.value("exact").toBool() ? "exact" : "unsupported");
			SetItem(m_pixel_watches, row, 4,
			        QString::number(JsonU64(event.value("timestamp_us")) / 1000.0, 'f', 3) + " ms");
			SetItem(m_pixel_watches, row, 5, QString::number(event.value("frame").toInt()));
			SetItem(m_pixel_watches, row, 6,
			        event.value("has_command").toBool()
			            ? event.value("command_kind").toString() + " #" +
			                  QString::number(event.value("command_index").toInt())
			            : QString());
			SetItem(m_pixel_watches, row, 7, event.value("action").toString());
			SetItem(m_pixel_watches, row, 8, ResourceIdentity(event));
			SetItem(m_pixel_watches, row, 9, ResourceShader(event));
			SetItem(m_pixel_watches, row, 10, ResourceOwner(event));
			SetItem(m_pixel_watches, row, 11, hit.value("status").toString());
		}
	}
	m_pixel_watches->verticalScrollBar()->setValue(
	    (std::min)(vertical, m_pixel_watches->verticalScrollBar()->maximum()));
	m_pixel_watches->horizontalScrollBar()->setValue(
	    (std::min)(horizontal, m_pixel_watches->horizontalScrollBar()->maximum()));
}

void MainWindow::RefreshGpuBreakConditions() {
	if (m_socket->state() != QLocalSocket::ConnectedState) return;
	FillGpuBreakConditions(Request("gpu_break_conditions"));
}

void MainWindow::FillGpuBreakConditions(const QJsonObject& response) {
	if (!response.value("ok").toBool()) {
		if (!response.value("error").toString().isEmpty())
			m_resource_trace_summary->setText(response.value("error").toString());
		return;
	}
	TableUpdateGuard update(m_gpu_break_conditions);
	const auto vertical = m_gpu_break_conditions->verticalScrollBar()->value();
	const auto horizontal = m_gpu_break_conditions->horizontalScrollBar()->value();
	const auto conditions = response.value("conditions").toArray();
	m_gpu_break_conditions->setRowCount(conditions.size());
	for (int row = 0; row < conditions.size(); row++) {
		const auto condition = conditions[row].toObject();
		const auto id = JsonU64(condition.value("id"));
		const auto value = JsonU64(condition.value("value"));
		SetItem(m_gpu_break_conditions, row, 0, QString::number(id));
		m_gpu_break_conditions->item(row, 0)->setData(Qt::UserRole, QVariant::fromValue(id));
		SetItem(m_gpu_break_conditions, row, 1, condition.value("kind").toString());
		SetItem(m_gpu_break_conditions, row, 2,
		        condition.value("kind").toString() == "pixel" ? QString::number(value)
		                                                          : Hex(value));
		SetItem(m_gpu_break_conditions, row, 3, condition.value("action").toString());
		SetItem(m_gpu_break_conditions, row, 4,
		        condition.value("enabled").toBool()
		            ? (condition.value("one_shot").toBool() ? "enabled, one shot" : "enabled")
		            : "triggered / disabled");
		SetItem(m_gpu_break_conditions, row, 5,
		        QString::number(JsonU64(condition.value("hits"))));
		SetItem(m_gpu_break_conditions, row, 6, condition.value("last_reason").toString());
	}
	m_gpu_break_conditions->verticalScrollBar()->setValue(
	    (std::min)(vertical, m_gpu_break_conditions->verticalScrollBar()->maximum()));
	m_gpu_break_conditions->horizontalScrollBar()->setValue(
	    (std::min)(horizontal, m_gpu_break_conditions->horizontalScrollBar()->maximum()));
}

void MainWindow::RefreshResourcePreview(bool refresh) {
	if (m_preview_address == 0 || m_socket->state() != QLocalSocket::ConnectedState) return;
	FillResourcePreview(Request("resource_preview", {{"address", QString::number(m_preview_address)},
	                                                 {"size", QString::number(m_preview_size)},
	                                                 {"refresh", refresh}}, 5000));
}

void MainWindow::FillResourcePreview(const QJsonObject& response) {
	if (!response.value("ok").toBool()) return;
	QLabel* preview_info = nullptr;
	QLabel* preview_image = nullptr;
	switch (m_preview_target) {
		case PreviewTarget::Shader:
			preview_info = m_shader_resource_preview_info;
			preview_image = m_shader_resource_preview;
			break;
		case PreviewTarget::ResourceTrace:
			preview_info = m_resource_trace_preview_info;
			preview_image = m_resource_trace_preview;
			break;
		case PreviewTarget::ResourceHistory:
			preview_info = m_resource_preview_info;
			preview_image = m_resource_preview;
			break;
	}
	const auto status = response.value("status").toString();
	if (status == "pending") {
		preview_info->setText("GPU readback pending…");
		return;
	}
	if (status != "ready") {
		preview_info->setText(response.value("error").toString());
		preview_image->setText("Preview unavailable");
		preview_image->setPixmap({});
		return;
	}
	const auto width = response.value("width").toInt();
	const auto height = response.value("height").toInt();
	const auto bytes = QByteArray::fromBase64(response.value("rgba_base64").toString().toLatin1());
	if (width <= 0 || height <= 0 || bytes.size() != width * height * 4) return;
	const QImage image(reinterpret_cast<const uchar*>(bytes.constData()), width, height,
	                   QImage::Format_RGBA8888);
	preview_image->setPixmap(QPixmap::fromImage(image.copy()).scaled(
	    preview_image->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	const auto total_pixels = JsonU64(response.value("total_pixels"));
	const auto non_finite_pixels = JsonU64(response.value("non_finite_pixels"));
	const auto zero_pixels = JsonU64(response.value("zero_pixels"));
	const auto non_finite_components = JsonU64(response.value("non_finite_components"));
	const auto non_finite_percent =
	    total_pixels == 0 ? 0.0 : 100.0 * static_cast<double>(non_finite_pixels) / total_pixels;
	preview_info->setText(
	    QString("%1×%2 → %3×%4  |  host format %5  |  hash %6  |  zero %7/%8  |  NaN/Inf %9 pixels (%10%, %11 components)")
	        .arg(response.value("source_width").toInt()).arg(response.value("source_height").toInt())
	        .arg(width).arg(height).arg(response.value("host_format").toInt())
	        .arg(Hex(JsonU64(response.value("content_hash"))))
	        .arg(zero_pixels).arg(total_pixels).arg(non_finite_pixels)
	        .arg(non_finite_percent, 0, 'f', 3).arg(non_finite_components));
}

void MainWindow::RefreshIo() {
	if (m_socket->state() != QLocalSocket::ConnectedState) return;
	FillIo(Request("io_history", {{"filter", m_io_filter->text()}, {"limit", 1500}}));
	FillIoFiles(Request("io_files", {{"filter", m_io_filter->text()}, {"limit", 4096}}));
}

void MainWindow::FillIo(const QJsonObject& response) {
	TableUpdateGuard update(m_io_events);
	const auto values = response.value("events").toArray();
	const auto vertical_position = m_io_events->verticalScrollBar()->value();
	const auto horizontal_position = m_io_events->horizontalScrollBar()->value();
	m_io_events->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_io_events, row, 0,
		        QString::number(JsonU64(value.value("timestamp_us")) / 1000.0, 'f', 3) + " ms");
		SetItem(m_io_events, row, 1, value.value("operation").toString());
		SetItem(m_io_events, row, 2, QString::number(value.value("descriptor").toInt()));
		SetItem(m_io_events, row, 3, value.value("guest_path").toString());
		SetItem(m_io_events, row, 4, value.value("host_path").toString());
		SetItem(m_io_events, row, 5, QString::number(value.value("offset").toVariant().toLongLong()));
		SetItem(m_io_events, row, 6, QString::number(JsonU64(value.value("requested"))));
		SetItem(m_io_events, row, 7, QString::number(value.value("result").toVariant().toLongLong()));
		SetItem(m_io_events, row, 8, QString::number(value.value("thread").toInt()));
		SetItem(m_io_events, row, 9, value.value("module").toString());
		SetItem(m_io_events, row, 10, Hex(JsonU64(value.value("caller"))));
	}
	// Periodic refreshes must never chase the newest event while the user is inspecting an older
	// file operation. Keep both scrollbars exactly where the user left them, including on the
	// initial fill (position zero).
	const auto vertical_maximum = m_io_events->verticalScrollBar()->maximum();
	const auto horizontal_maximum = m_io_events->horizontalScrollBar()->maximum();
	m_io_events->verticalScrollBar()->setValue(
	    vertical_position < vertical_maximum ? vertical_position : vertical_maximum);
	m_io_events->horizontalScrollBar()->setValue(
	    horizontal_position < horizontal_maximum ? horizontal_position : horizontal_maximum);
}

void MainWindow::FillIoFiles(const QJsonObject& response) {
	TableUpdateGuard update(m_io_files);
	const auto values = response.value("files").toArray();
	const auto vertical_position = m_io_files->verticalScrollBar()->value();
	const auto horizontal_position = m_io_files->horizontalScrollBar()->value();
	m_io_files->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		SetItem(m_io_files, row, 0, value.value("guest_path").toString());
		SetItem(m_io_files, row, 1, value.value("host_path").toString());
		SetItem(m_io_files, row, 2, QString::number(JsonU64(value.value("opens"))));
		SetItem(m_io_files, row, 3, QString::number(JsonU64(value.value("closes"))));
		SetItem(m_io_files, row, 4, QString::number(JsonU64(value.value("reads"))));
		SetItem(m_io_files, row, 5, QString::number(JsonU64(value.value("writes"))));
		SetItem(m_io_files, row, 6, QString::number(JsonU64(value.value("seeks"))));
		SetItem(m_io_files, row, 7, QString::number(JsonU64(value.value("stats"))));
		SetItem(m_io_files, row, 8, QString::number(JsonU64(value.value("bytes_read"))));
		SetItem(m_io_files, row, 9, QString::number(JsonU64(value.value("bytes_written"))));
		SetItem(m_io_files, row, 10,
		        QString::number(JsonU64(value.value("last_timestamp_us")) / 1000.0, 'f', 3) + " ms");
		SetItem(m_io_files, row, 11, value.value("module").toString());
	}
	const auto vertical_maximum = m_io_files->verticalScrollBar()->maximum();
	const auto horizontal_maximum = m_io_files->horizontalScrollBar()->maximum();
	m_io_files->verticalScrollBar()->setValue(
	    vertical_position < vertical_maximum ? vertical_position : vertical_maximum);
	m_io_files->horizontalScrollBar()->setValue(
	    horizontal_position < horizontal_maximum ? horizontal_position : horizontal_maximum);
}

void MainWindow::ReadMemory() {
	bool size_ok = false;
	const auto size = m_memory_size->text().toUInt(&size_ok, 0);
	if (!size_ok) return;
	const auto response = Request("memory_read", {{"address", m_memory_address->text()},
	                                              {"size", static_cast<int>(size)}});
	if (!response.value("ok").toBool()) {
		m_memory->setPlainText(response.value("error").toString());
		return;
	}
	const auto bytes = QByteArray::fromHex(response.value("bytes").toString().toLatin1());
	const auto base = JsonU64(response.value("address"));
	QString text;
	for (int offset = 0; offset < bytes.size(); offset += 16) {
		text += QString("%1  ").arg(base + static_cast<quint64>(offset), 16, 16, QLatin1Char('0'));
		QString ascii;
		for (int column = 0; column < 16; column++) {
			if (offset + column < bytes.size()) {
				const auto byte = static_cast<unsigned char>(bytes[offset + column]);
				text += QString("%1 ").arg(byte, 2, 16, QLatin1Char('0'));
				ascii += byte >= 32 && byte < 127 ? QChar(byte) : QChar('.');
			} else {
				text += "   ";
			}
		}
		text += " " + ascii + '\n';
	}
	m_memory->setPlainText(text);
}

void MainWindow::FillFrame(const QJsonObject& response) {
	TableUpdateGuard update(m_frame);
	const auto values = response.value("commands").toArray();
	m_frame->setRowCount(values.size());
	for (int row = 0; row < values.size(); row++) {
		const auto value = values[row].toObject();
		const auto groups = value.value("groups").toArray();
		const auto detail = value.value("kind").toString() == "dispatch"
		                        ? QString("%1 × %2 × %3").arg(groups[0].toInt()).arg(groups[1].toInt()).arg(groups[2].toInt())
		                        : QString("%1 × %2").arg(value.value("count").toInt()).arg(value.value("instances").toInt());
		SetItem(m_frame, row, 0, QString::number(value.value("index").toInt()));
		SetItem(m_frame, row, 1, value.value("kind").toString());
		SetItem(m_frame, row, 2, QString::number(JsonU64(value.value("submit"))));
		m_frame->item(row, 2)->setData(Qt::UserRole,
		                                QVariant::fromValue(JsonU64(value.value("submit"))));
		SetItem(m_frame, row, 3, detail);
		SetItem(m_frame, row, 4, Hex(JsonU64(value.value("vs"))));
		SetItem(m_frame, row, 5, Hex(JsonU64(value.value("ps"))));
		SetItem(m_frame, row, 6, Hex(JsonU64(value.value("cs"))));
	}
}

void MainWindow::SetConnectionState(const QString& text, bool connected) {
	m_status->setText(text);
	m_connect->setText(connected ? "Detach" : "Attach");
	m_pause->setEnabled(connected);
	m_continue->setEnabled(connected);
	m_step_into->setEnabled(connected);
	m_step_over->setEnabled(connected);
	m_step_out->setEnabled(connected);
	if (!connected) m_timer->stop();
}
