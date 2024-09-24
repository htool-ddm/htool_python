#ifndef HTOOL_PYTHON_MISC_TESTING_CPP
#define HTOOL_PYTHON_MISC_TESTING_CPP
#include <htool/misc/logger.hpp>

void test_logger() {
    htool::Logger::get_instance().log(htool::LogLevel::CRITICAL, "Critical message");
    htool::Logger::get_instance().log(htool::LogLevel::ERROR, "Error message");
    htool::Logger::get_instance().log(htool::LogLevel::WARNING, "Warning message");
    htool::Logger::get_instance().log(htool::LogLevel::DEBUG, "Debug message");
    htool::Logger::get_instance().log(htool::LogLevel::INFO, "Info message");
}

#endif
