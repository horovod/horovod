## Contributing to Horovod

**Thanks for taking the time to contribute!**

Refer to the following guidelines to contribute new functionality or bug fixes to Horovod:
1. Use [autopep8](https://github.com/hhatto/autopep8) to format the Python code.
2. Use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format C++ code.
3. Add unit tests for any new code you write.
4. Run unit tests in both CPU and GPU environments.

### Charter

You can find Horovod Charter [here](https://wiki.lfai.foundation/download/attachments/7733301/Horovod%20Project%20Technical%20Charter%2012-22-2018%20FINAL.pdf?version=1&modificationDate=1558389484000&api=v2)

### Technical Steering Committee

Horovod development is governed by the Horovod Technical Steering Committee (TSC). The TSC consists of voting and
non-voting members, in addition to a chairman responsible for running TSC meetings, setting the meeting agenda, and
calling votes on proposals.

Current voting members of the Horovod TSC:
* Alex Sergeev (@alsrgv)
* Travis Addair (@tgaddair) - Chairman
* Can Karakus (@karakusc)
* Josh Romero (@romerojosh)
* Jaliya Ekanayake (@jaliyae)

Current non-voting members of the Horovod TSC:
* Yuxi Hu (@yuxihu)
* Lin Yuan (@apeforest)
* Todd Mytkowicz (@klipto)
* Emad Barsoum (@ebarsoum)
* Kaarthik Sivashanmugam (@skaarthik)
* Nicolas Castet (@nvcastet)

Non-voting members of the TSC ("maintainers") have commit access to the Horovod GitHub repository, and take part in the
standing TSC meetings and mailing lists.

The Horovod TSC meets monthly and publishes meeting notes via a [mailing list](https://lists.lfai.foundation/g/horovod-tsc).
This mailing list can also be utilized to reach out to the TSC.  Major decisions regarding the technical directions of
the Horovod project will be brought before the TSC for discussion, with an accompanying proposal document termed an RFC
(Request for Comments).

Technical decisions made by the TSC should be unanimous, with each voting and non-voting member either agreeing to the
proposal or abstaining for it to pass.  If consensus cannot be reached, then the proposal is to be put to a vote
among the voting members of the TSC, at which point a majority of the voting TSC must agree to the proposal for it to
pass.

Decisions to add or change members of the TSC in either a voting or non-voting capacity are handled the same as other
proposals (without an RFC): an attempt is made to reach a unanimous decision among the entire TSC, followed by a vote
among the voting members if no consensus can be reached.

### Code of Conduct

Please be mindful of and adhere to the Linux Foundation's
[Code of Conduct](https://lfprojects.org/policies/code-of-conduct) when contributing to Horovod.
