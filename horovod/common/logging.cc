#include <chrono>

#include "logging.h"

namespace horovod {
namespace common {

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::GenerateLogMessage() {
  auto now = std::chrono::system_clock::now();
  auto as_time_t = std::chrono::system_clock::to_time_t(now);

  auto duration = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
  auto micros_remainder = std::chrono::duration_cast<std::chrono::microseconds>(duration - seconds);

  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&as_time_t));

  fprintf(stdout, "[%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder,
          "TDIWEF"[severity_], fname_, line_, str().c_str());
}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) GenerateLogMessage();
}


LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}
LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  abort();
}

void LogString(const char* fname, int line, int severity,
               const std::string& message) {
  LogMessage(fname, line, severity) << message;
}

int LogLevelStrToInt(const char* env_var_val) {
  if (env_var_val == nullptr) {
    return 0;
  }
  std::string min_log_level(env_var_val);
  std::istringstream ss(min_log_level);
  int level;
  if (!(ss >> level)) {
    // Invalid log level setting, set level to default (0)
    level = 0;
  }

  return level;
}

int MinLogLevelFromEnv() {
  const char* env_var_val = getenv("HOROVOD_LOG_LEVEL");
  if (env_var_val == nullptr) {
    // default to WARN
    return 3;
  }
  return LogLevelStrToInt(env_var_val);
}


}
}