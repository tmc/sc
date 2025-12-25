// Package scriptcmp provides a cmp command for rsc.io/script tests
// with support for updating golden files in txtar archives.
package scriptcmp

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/txtar"
	"rsc.io/script"
)

// Updater collects file updates to apply to txtar archives.
type Updater struct {
	t           *testing.T
	updates     []update
	testdataDir string
	updateMode  bool
}

type update struct {
	txtarPath string
	fileName  string
	content   []byte
}

// NewUpdater creates an Updater for the given testdata directory.
func NewUpdater(t *testing.T, testdataDir string, updateMode bool) *Updater {
	t.Helper()
	u := &Updater{t: t, testdataDir: testdataDir, updateMode: updateMode}
	if updateMode {
		t.Cleanup(func() {
			if err := u.apply(); err != nil {
				t.Errorf("failed to apply updates: %v", err)
			}
		})
	}
	return u
}

// Cmd returns a cmp command using this updater.
func (u *Updater) Cmd() script.Cmd {
	return Cmd(u, u.updateMode)
}

func (u *Updater) apply() error {
	byFile := make(map[string][]update)
	for _, upd := range u.updates {
		byFile[upd.txtarPath] = append(byFile[upd.txtarPath], upd)
	}
	for txtarPath, updates := range byFile {
		if err := applyUpdates(txtarPath, updates); err != nil {
			return err
		}
	}
	return nil
}

func applyUpdates(txtarPath string, updates []update) error {
	ar, err := txtar.ParseFile(txtarPath)
	if err != nil {
		return fmt.Errorf("parse %s: %w", txtarPath, err)
	}

	for _, upd := range updates {
		found := false
		for i := range ar.Files {
			if ar.Files[i].Name == upd.fileName {
				ar.Files[i].Data = upd.content
				found = true
				break
			}
		}
		if !found {
			ar.Files = append(ar.Files, txtar.File{Name: upd.fileName, Data: upd.content})
		}
	}

	return os.WriteFile(txtarPath, txtar.Format(ar), 0644)
}

// Cmd returns a script.Cmd that compares files or stdout/stderr.
func Cmd(updater *Updater, updateMode bool) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "compare file contents (or stdout/stderr)",
			Args:    "actual expected",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) != 2 {
				return nil, script.ErrUsage
			}

			got, err := readArg(s, args[0])
			if err != nil {
				return nil, err
			}
			want, err := readArg(s, args[1])
			if err != nil {
				return nil, err
			}

			if bytes.Equal(got, want) {
				return nil, nil
			}

			if updateMode && updater != nil && args[1] != "stdout" && args[1] != "stderr" {
				txtarPath := findTxtarSource(s.Getwd(), updater.testdataDir)
				if txtarPath != "" {
					updater.updates = append(updater.updates, update{
						txtarPath: txtarPath,
						fileName:  args[1],
						content:   got,
					})
					return nil, nil
				}
			}

			return nil, fmt.Errorf("%s and %s differ:\n%s", args[0], args[1], cmp.Diff(string(want), string(got)))
		},
	)
}

func readArg(s *script.State, name string) ([]byte, error) {
	switch name {
	case "stdout":
		return []byte(s.Stdout()), nil
	case "stderr":
		return []byte(s.Stderr()), nil
	default:
		return os.ReadFile(s.Path(name))
	}
}

func findTxtarSource(workDir, testdataDir string) string {
	for _, part := range strings.Split(workDir, string(filepath.Separator)) {
		if !strings.HasPrefix(part, "Test") {
			continue
		}
		name := strings.TrimPrefix(part, "Test")
		for len(name) > 0 && name[len(name)-1] >= '0' && name[len(name)-1] <= '9' {
			name = name[:len(name)-1]
		}
		for i := 0; i < len(name); i++ {
			candidate := filepath.Join(testdataDir, name[i:]+".txt")
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
		}
	}
	return ""
}
