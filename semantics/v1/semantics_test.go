package semantics

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/tmc/sc"
)

func TestStatechart_Children(t *testing.T) {
	tests := []struct {
		name    string
		chart   *Statechart
		state   StateLabel
		want    []StateLabel
		wantErr bool
	}{
		{"invalid", exampleStatechart1, StateLabel("this state does not exist"), nil, true},
		{"valid but not toplevel", exampleStatechart1, StateLabel("Turnstile Control"), nil, false},
		{"Off", exampleStatechart1, StateLabel("Off"), nil, false},
		{"On", exampleStatechart1, StateLabel("On"), nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.Children(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Children() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(got, tt.want) {
				t.Log(cmp.Diff(got, tt.want))
			}
		})
	}
}

func TestStatechart_findState(t *testing.T) {
	type fields struct {
		Statechart *sc.Statechart
	}
	type args struct {
		label StateLabel
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *sc.State
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Statechart{
				Statechart: tt.fields.Statechart,
			}
			got, err := s.findState(tt.args.label)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.findState() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Statechart.findState() = %v, want %v", got, tt.want)
			}
		})
	}
}
