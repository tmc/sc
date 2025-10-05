// Package version provides version information for the sc project.
package version

import (
	"fmt"
	"runtime/debug"
)

var (
	// Version is the semantic version (set by build flags or defaults to dev)
	Version = "dev"
	// GitCommit is the git commit hash (set by build flags or detected from VCS)
	GitCommit = ""
	// BuildDate is the build timestamp (set by build flags)
	BuildDate = ""
)

// Info returns formatted version information.
func Info() string {
	commit := GitCommit
	if commit == "" {
		commit = getVCSRevision()
	}

	info := fmt.Sprintf("sc version %s", Version)
	if commit != "" {
		info += fmt.Sprintf(" (commit: %s)", commit)
	}
	if BuildDate != "" {
		info += fmt.Sprintf(" (built: %s)", BuildDate)
	}
	return info
}

// getVCSRevision attempts to get the VCS revision from runtime/debug build info.
func getVCSRevision() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" {
				if len(setting.Value) > 7 {
					return setting.Value[:7]
				}
				return setting.Value
			}
		}
	}
	return ""
}
