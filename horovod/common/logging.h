#ifndef HOROVOD_LOGGING_H
#define HOROVOD_LOGGING_H

#include <sstream>
#include <string>

namespace horovod {
namespace common {

enum class LogLevel {
  TRACE, DEBUG, INFO, WARNING, ERROR, FATAL
};

#define LOG_LEVELS "TDIWEF"

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, LogLevel severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage(bool log_time);

 private:
  const char* fname_;
  int line_;
  LogLevel severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _HVD_LOG_TRACE \
  LogMessage(__FILE__, __LINE__, LogLevel::TRACE)
#define _HVD_LOG_DEBUG \
  LogMessage(__FILE__, __LINE__, LogLevel::DEBUG)
#define _HVD_LOG_INFO \
  LogMessage(__FILE__, __LINE__, LogLevel::INFO)
#define _HVD_LOG_WARNING \
  LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _HVD_LOG_ERROR \
  LogMessage(__FILE__, __LINE__, LogLevel::ERROR)
#define _HVD_LOG_FATAL \
  LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _HVD_LOG_##severity

#define _LOG_RANK(severity, rank) _HVD_LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();

}
}

#endif // HOROVOD_LOGGING_H
