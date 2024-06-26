package semantics

import (
	"fmt"

	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
	"google.golang.org/protobuf/types/known/structpb"
)

func HandleEvent(machine *sc.Machine, event string) (bool, error) {
	for _, transition := range machine.Statechart.Transitions {
		if transition.Event == event && slices.Contains(transition.From, machine.Configuration.States[0].Label) {
			// Execute transition
			machine.Configuration.States[0].Label = transition.To[0]

			// Increment count only for handled events
			if machine.Context != nil && machine.Context.Fields != nil {
				if countValue, exists := machine.Context.Fields["count"]; exists {
					if count, ok := countValue.GetKind().(*structpb.Value_NumberValue); ok {
						newCount := structpb.NewNumberValue(count.NumberValue + 1)
						machine.Context.Fields["count"] = newCount
					} else {
						return false, fmt.Errorf("count field is not a number")
					}
				}
			}

			return true, nil
		}
	}
	return false, nil
}
