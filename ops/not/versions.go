package not

import "github.com/advancedclimatesystems/gonnx/ops"

var notVersions = ops.OperatorVersions{
	1: ops.NewOperatorConstructor(newNot, 1, notTypeConstraints),
}

func GetNotVersions() ops.OperatorVersions {
	return notVersions
}