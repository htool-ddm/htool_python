#ifndef HTOOL_PYTHON_MISC_LOGGER_CPP
#define HTOOL_PYTHON_MISC_LOGGER_CPP

#include <htool/misc/logger.hpp>
#include <pybind11/pybind11.h>

// Delegates formatting and filtering to python logging module
// mapping python logging level to Htool logging level

class PythonLoggerWriter : public IObjectWriter {

  public:
    void write(LogLevel logging_level, const std::string &message) override {
        py::object logging_module = py::module::import("logging");
        py::object logger         = logging_module.attr("getLogger")("Htool");
        switch (logging_level) {
        case LogLevel::CRITICAL:
            logger.attr("critical")(message);
            break;
        case LogLevel::ERROR:
            logger.attr("error")(message);
            break;
        case LogLevel::WARNING:
            logger.attr("warning")(message);
            break;
        case LogLevel::DEBUG:
            logger.attr("debug")(message);
            break;
        case LogLevel::INFO:
            logger.attr("info")(message);
            break;
        default:
            break;
        }
    }

    void set_log_level(LogLevel log_level) override {
    }
};

#endif
