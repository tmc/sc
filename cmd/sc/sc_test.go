package main

import (
	"context"
	"flag"
	"os/exec"
	"path"
	"testing"

	"github.com/tmc/sc/cmd/sc/internal/scriptcmp"
	"rsc.io/script"
	"rsc.io/script/scripttest"
)

var update = flag.Bool("update", false, "update golden files in txtar archives")

func TestSC(t *testing.T) {
	scPath := path.Join(t.TempDir(), "sc")
	if err := exec.Command("go", "build", "-o", scPath, ".").Run(); err != nil {
		t.Fatalf("failed to build sc: %v", err)
	}

	updater := scriptcmp.NewUpdater(t, "testdata", *update)

	cmds := scripttest.DefaultCmds()
	delete(cmds, "exec") // Security: don't allow arbitrary exec
	cmds["sc"] = script.Program(scPath, nil, 0)
	cmds["stdin-sc"] = stdinScCmd(scPath)
	cmds["cmp"] = updater.Cmd()

	engine := &script.Engine{
		Cmds:  cmds,
		Conds: scripttest.DefaultConds(),
	}
	scripttest.Test(t, context.Background(), engine, []string{}, "testdata/*.txt")
}

// stdinScCmd runs sc with the first arg redirected to stdin.
func stdinScCmd(scPath string) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "run sc with file as stdin",
			Args:    "file [args...]",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) < 1 {
				return nil, script.ErrUsage
			}
			cmd := scPath
			for _, arg := range args[1:] {
				cmd += " " + arg
			}
			cmd += " < " + s.Path(args[0])
			return script.Program("sh", nil, 0).Run(s, "-c", cmd)
		},
	)
}
