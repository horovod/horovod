#include <sstream>
#include <string>

namespace horovod {
namespace common {

const int TRACE = 0;
const int DEBUG = 1;
const int INFO = 2;
const int WARNING = 3;
const int ERROR = 4;
const int FATAL = 5;

void LogString(const char* fname, int line, int severity,
               const std::string& message);

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

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
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _HVD_LOG_TRACE \
  LogMessage(__FILE__, __LINE__, TRACE)
#define _HVD_LOG_DEBUG \
  LogMessage(__FILE__, __LINE__, DEBUG)
#define _HVD_LOG_INFO \
  LogMessage(__FILE__, __LINE__, INFO)
#define _HVD_LOG_WARNING \
  LogMessage(__FILE__, __LINE__, WARNING)
#define _HVD_LOG_ERROR \
  LogMessage(__FILE__, __LINE__, ERROR)
#define _HVD_LOG_FATAL \
  LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _HVD_LOG_##severity

#define _LOG_RANK(severity, rank) _HVD_LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

int MinLogLevelFromEnv();

}
}
