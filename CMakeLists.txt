project (gpt C)

set (SRC host/main.c)

add_executable (${PROJECT_NAME} ${SRC})

find_package(Threads REQUIRED)
target_include_directories(${PROJECT_NAME}
			   PRIVATE ta/include
			   PRIVATE host/include
			   PRIVATE include)

target_link_libraries (${PROJECT_NAME} PRIVATE teec -lm Threads::Threads)

install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
