import SwiftUI

struct StatechartViewModelKey: FocusedValueKey {
    typealias Value = StatechartViewModel
}

extension FocusedValues {
    var statechartViewModel: StatechartViewModel? {
        get { self[StatechartViewModelKey.self] }
        set { self[StatechartViewModelKey.self] = newValue }
    }
}
