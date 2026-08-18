find_package(Qt6 COMPONENTS Network Widgets REQUIRED)

add_executable(kyty_debugger WIN32 main.cpp mainWindow.cpp mainWindow.h)
target_link_libraries(kyty_debugger PRIVATE Qt6::Network Qt6::Widgets)
set_target_properties(kyty_debugger PROPERTIES AUTOMOC ON OUTPUT_NAME "kyty_debugger")

if(WIN32)
	find_program(KYTY_DEBUGGER_WINDEPLOYQT NAMES windeployqt PATHS "${Qt6_DIR}/../../../bin")
	if(NOT KYTY_DEBUGGER_WINDEPLOYQT)
		message(FATAL_ERROR "Could not find windeployqt")
	endif()
	add_custom_command(TARGET kyty_debugger POST_BUILD
		COMMAND ${KYTY_DEBUGGER_WINDEPLOYQT} $<TARGET_FILE:kyty_debugger>
			--no-compiler-runtime --no-system-d3d-compiler --no-system-dxc-compiler
			--no-opengl-sw --no-translations
		COMMENT "Deploying external debugger Qt runtime"
		VERBATIM)
endif()

install(TARGETS kyty_debugger DESTINATION .)

if(WIN32)
	install(CODE "
		execute_process(COMMAND ${KYTY_DEBUGGER_WINDEPLOYQT}
			\"\${CMAKE_INSTALL_PREFIX}/kyty_debugger.exe\"
			--no-compiler-runtime --no-system-d3d-compiler --no-system-dxc-compiler
			--no-opengl-sw --no-translations)")
endif()
