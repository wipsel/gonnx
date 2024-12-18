package greaterorequal

import "github.com/advancedclimatesystems/gonnx/ops"

var greaterOrEqualVersions = ops.OperatorVersions{
	12: ops.NewOperatorConstructor(newGreaterOrEqual, 12, greaterOrEqualTypeConstraints),
}

func GetGreaterOrEqualVersions() ops.OperatorVersions {
	return greaterOrEqualVersions
}