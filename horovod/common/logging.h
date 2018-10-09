#include <limits>
#include <sstream>

namespace horovod {
namespace common {
const int DEBUG = 0;
const int INFO = 1;
const int WARNING = 2;
const int ERROR = 3;
const int FATAL = 4;
const int NUM_SEVERITIES = 5;

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64 MinVLogLevel();

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) TF_ATTRIBUTE_COLD;
  TF_ATTRIBUTE_NORETURN ~LogMessageFatal();
};

#define _TF_LOG_INFO \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::INFO)
#define _TF_LOG_WARNING \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::WARNING)
#define _TF_LOG_ERROR \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::ERROR)
#define _TF_LOG_FATAL \
  ::tensorflow::internal::LogMessageFatal(__FILE__, __LINE__)

#define LOG(severity) _TF_LOG_##severity

#define _TF_LOG_QFATAL _TF_LOG_FATAL

int64 MinLogLevelFromEnv();

int64 MinVLogLevelFromEnv();

}
}
