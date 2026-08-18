#ifndef KYTY_DEBUGGER_APP_MAIN_WINDOW_H_
#define KYTY_DEBUGGER_APP_MAIN_WINDOW_H_

#include <QJsonObject>
#include <QMainWindow>
#include <QString>
#include <QVector>

class QComboBox;
class QCheckBox;
class QLabel;
class QLineEdit;
class QLocalSocket;
class QPlainTextEdit;
class QPushButton;
class QTableWidget;
class QTabWidget;
class QTimer;
class QWidget;
class ResourceAliasMapWidget;

class MainWindow final: public QMainWindow {
public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

private:
	struct SessionDescriptor {
		quint64 pid = 0;
		QString endpoint;
		QString token;
	};

	void        RefreshSessions();
	void        ConnectSelected();
	void        Disconnect();
	void        RefreshData();
	QJsonObject Request(const QString& command, const QJsonObject& arguments = {},
	                    int timeout_ms = 1000);
	void        FillSummary(const QJsonObject& response);
	void        FillThreads(const QJsonObject& response);
	void        FillModules(const QJsonObject& response);
	void        FillShaders(const QJsonObject& response);
	void        FillFrame(const QJsonObject& response);
	void        FillStoppedDetails(const QJsonObject& response);
	void        FillBreakpoints(const QJsonObject& response);
	void        RefreshSymbols();
	void        FillSymbols(const QJsonObject& response);
	void        LoadShader(quint64 hash);
	void        FillShaderDetails(const QJsonObject& response);
	void        FillResourceHistory(const QJsonObject& response);
	void        RefreshResourceTrace();
	void        FillResourceTrace(const QJsonObject& response);
	void        TraceRange(quint64 address, quint64 size);
	void        RefreshPixelWatches();
	void        FillPixelWatches(const QJsonObject& response);
	void        RefreshGpuBreakConditions();
	void        FillGpuBreakConditions(const QJsonObject& response);
	void        FillRenderDocStatus(const QJsonObject& response);
	void        RefreshResourcePreview(bool refresh = false);
	void        FillResourcePreview(const QJsonObject& response);
	void        RefreshIo();
	void        FillIo(const QJsonObject& response);
	void        FillIoFiles(const QJsonObject& response);
	void        ReadMemory();
	void        SetConnectionState(const QString& text, bool connected);

	QComboBox*    m_sessions = nullptr;
	QPushButton*  m_connect = nullptr;
	QPushButton*  m_pause = nullptr;
	QPushButton*  m_continue = nullptr;
	QPushButton*  m_step_into = nullptr;
	QPushButton*  m_step_over = nullptr;
	QPushButton*  m_step_out = nullptr;
	QLabel*       m_status = nullptr;
	QLabel*       m_summary = nullptr;
	QTableWidget* m_threads = nullptr;
	QTableWidget* m_modules = nullptr;
	QTableWidget* m_symbols = nullptr;
	QTableWidget* m_shaders = nullptr;
	QTableWidget* m_frame = nullptr;
	QTableWidget* m_registers = nullptr;
	QTableWidget* m_callstack = nullptr;
	QTableWidget* m_disassembly = nullptr;
	QTableWidget* m_breakpoints = nullptr;
	QTableWidget* m_resources = nullptr;
	QTableWidget* m_resource_history = nullptr;
	QTableWidget* m_resource_trace_aliases = nullptr;
	QTableWidget* m_resource_trace_events = nullptr;
	QTableWidget* m_pixel_watches = nullptr;
	QTableWidget* m_gpu_break_conditions = nullptr;
	QTableWidget* m_io_events = nullptr;
	QTableWidget* m_io_files = nullptr;
	QLineEdit*    m_symbol_filter = nullptr;
	QLineEdit*    m_io_filter = nullptr;
	QComboBox*    m_symbol_module = nullptr;
	QLineEdit*    m_breakpoint_location = nullptr;
	QLineEdit*    m_memory_address = nullptr;
	QLineEdit*    m_memory_size = nullptr;
	QLineEdit*    m_resource_trace_address = nullptr;
	QLineEdit*    m_resource_trace_size = nullptr;
	QLineEdit*    m_pixel_watch_x = nullptr;
	QLineEdit*    m_pixel_watch_y = nullptr;
	QComboBox*    m_gpu_break_kind = nullptr;
	QLineEdit*    m_gpu_break_value = nullptr;
	QLineEdit*    m_gpu_break_action = nullptr;
	QCheckBox*    m_gpu_break_one_shot = nullptr;
	QPlainTextEdit* m_memory = nullptr;
	QPlainTextEdit* m_shader_isa = nullptr;
	QPlainTextEdit* m_shader_ir = nullptr;
	QPlainTextEdit* m_shader_spirv = nullptr;
	QLabel*         m_shader_title = nullptr;
	QLabel*         m_shader_resource_preview = nullptr;
	QLabel*         m_shader_resource_preview_info = nullptr;
	QLabel*         m_resource_preview = nullptr;
	QLabel*         m_resource_preview_info = nullptr;
	QLabel*         m_resource_trace_preview = nullptr;
	QLabel*         m_resource_trace_preview_info = nullptr;
	QLabel*         m_resource_trace_summary = nullptr;
	QLabel*         m_command_capture_status = nullptr;
	QLabel*         m_renderdoc_status = nullptr;
	QPushButton*    m_renderdoc_capture = nullptr;
	QTabWidget*     m_graphics_tabs = nullptr;
	QWidget*        m_resource_trace_page = nullptr;
	ResourceAliasMapWidget* m_resource_alias_map = nullptr;
	QLocalSocket* m_socket = nullptr;
	QTimer*       m_timer = nullptr;
	QTimer*       m_discovery = nullptr;
	QVector<SessionDescriptor> m_descriptors;
	QString       m_token;
	int           m_selected_thread = 0;
	quint64       m_selected_shader = 0;
	quint64       m_preview_address = 0;
	quint64       m_preview_size = 0;
	enum class PreviewTarget { ResourceHistory, Shader, ResourceTrace };
	PreviewTarget m_preview_target = PreviewTarget::ResourceHistory;
};

#endif // KYTY_DEBUGGER_APP_MAIN_WINDOW_H_
