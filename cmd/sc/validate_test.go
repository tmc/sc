package main

import (
	"bytes"
	"strings"
	"testing"
)

const undeclaredEventChart = `{
  "name": "x",
  "rootState": {
    "label": "root",
    "type": "STATE_TYPE_OR",
    "children": [
      {"label":"A","type":"STATE_TYPE_BASIC","isInitial":true},
      {"label":"B","type":"STATE_TYPE_BASIC"}
    ]
  },
  "events": [
    {"label":"KNOWN"}
  ],
  "transitions": [
    {"from":["A"],"to":["B"],"event":"UNKNOWN"}
  ]
}`

func TestCmdValidate_StrictByDefault(t *testing.T) {
	var stdout bytes.Buffer
	err := cmdValidate(nil, strings.NewReader(undeclaredEventChart), &stdout, &bytes.Buffer{})
	if err == nil {
		t.Fatal("cmdValidate() error = nil, want non-nil")
	}
	if !strings.Contains(stdout.String(), "INVALID:") {
		t.Fatalf("stdout %q does not contain INVALID", stdout.String())
	}
}

func TestCmdValidate_NonStrictAllowsUndeclaredEvent(t *testing.T) {
	var stdout bytes.Buffer
	err := cmdValidate([]string{"-strict=false"}, strings.NewReader(undeclaredEventChart), &stdout, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("cmdValidate() error = %v, want nil", err)
	}
	if !strings.Contains(stdout.String(), "VALID") {
		t.Fatalf("stdout %q does not contain VALID", stdout.String())
	}
}
