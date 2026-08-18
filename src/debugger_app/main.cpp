#include "mainWindow.h"

#include <QApplication>

int main(int argc, char* argv[]) {
	QApplication app(argc, argv);
	QApplication::setApplicationName("Kyty Debugger");
	QApplication::setOrganizationName("KytyPS5");

	MainWindow window;
	window.resize(1180, 720);
	window.show();
	return QApplication::exec();
}
