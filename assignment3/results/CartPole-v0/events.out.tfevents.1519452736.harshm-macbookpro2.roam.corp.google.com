       �K"	   �@��Abrain.Event:2�/�	y      $#*b	���@��A"��
d
oPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
a
a_discPlaceholder*
dtype0*
shape:���������*#
_output_shapes
:���������
`
advtgPlaceholder*
dtype0*
shape:���������*#
_output_shapes
:���������
�
Gpolicy_network/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB"      *
_output_shapes
:
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�7�*
_output_shapes
: 
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�7?*
_output_shapes
: 
�
Opolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformGpolicy_network/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*9
_class/
-+loc:@policy_network/fully_connected/weights
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/subSubEpolicy_network/fully_connected/weights/Initializer/random_uniform/maxEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes
: 
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/mulMulOpolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformEpolicy_network/fully_connected/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
Apolicy_network/fully_connected/weights/Initializer/random_uniformAddEpolicy_network/fully_connected/weights/Initializer/random_uniform/mulEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
&policy_network/fully_connected/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*9
_class/
-+loc:@policy_network/fully_connected/weights*
shared_name 
�
-policy_network/fully_connected/weights/AssignAssign&policy_network/fully_connected/weightsApolicy_network/fully_connected/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
�
+policy_network/fully_connected/weights/readIdentity&policy_network/fully_connected/weights*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
7policy_network/fully_connected/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
�
%policy_network/fully_connected/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
,policy_network/fully_connected/biases/AssignAssign%policy_network/fully_connected/biases7policy_network/fully_connected/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
�
*policy_network/fully_connected/biases/readIdentity%policy_network/fully_connected/biases*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
�
%policy_network/fully_connected/MatMulMatMulo+policy_network/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
&policy_network/fully_connected/BiasAddBiasAdd%policy_network/fully_connected/MatMul*policy_network/fully_connected/biases/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
�
#policy_network/fully_connected/ReluRelu&policy_network/fully_connected/BiasAdd*
T0*'
_output_shapes
:���������
�
Ipolicy_network/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB"      *
_output_shapes
:
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *:��*
_output_shapes
: 
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *:�?*
_output_shapes
: 
�
Qpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*;
_class1
/-loc:@policy_network/fully_connected_1/weights
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/subSubGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes
: 
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulMulQpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
Cpolicy_network/fully_connected_1/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
(policy_network/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
�
/policy_network/fully_connected_1/weights/AssignAssign(policy_network/fully_connected_1/weightsCpolicy_network/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
�
-policy_network/fully_connected_1/weights/readIdentity(policy_network/fully_connected_1/weights*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
9policy_network/fully_connected_1/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
�
'policy_network/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
�
.policy_network/fully_connected_1/biases/AssignAssign'policy_network/fully_connected_1/biases9policy_network/fully_connected_1/biases/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
�
,policy_network/fully_connected_1/biases/readIdentity'policy_network/fully_connected_1/biases*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
�
'policy_network/fully_connected_1/MatMulMatMul#policy_network/fully_connected/Relu-policy_network/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
(policy_network/fully_connected_1/BiasAddBiasAdd'policy_network/fully_connected_1/MatMul,policy_network/fully_connected_1/biases/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
e
#multinomial/Multinomial/num_samplesConst*
dtype0*
value	B :*
_output_shapes
: 
�
multinomial/MultinomialMultinomial(policy_network/fully_connected_1/BiasAdd#multinomial/Multinomial/num_samples*
seed2 *

seed *
T0*
output_dtype0	*'
_output_shapes
:���������
p
SqueezeSqueezemultinomial/Multinomial*
squeeze_dims
*
T0	*#
_output_shapes
:���������
o
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapea_disc*
out_type0*
T0*
_output_shapes
:
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits(policy_network/fully_connected_1/BiasAdda_disc*
T0*
Tlabels0*6
_output_shapes$
":���������:���������
�
NegNegGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:���������
?
Neg_1NegNeg*
T0*#
_output_shapes
:���������
F
mulMulNeg_1advtg*
T0*#
_output_shapes
:���������
R
gradients/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
j
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*#
_output_shapes
:���������
]
gradients/mul_grad/ShapeShapeNeg_1*
out_type0*
T0*
_output_shapes
:
_
gradients/mul_grad/Shape_1Shapeadvtg*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
b
gradients/mul_grad/mulMulgradients/Filladvtg*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
d
gradients/mul_grad/mul_1MulNeg_1gradients/Fill*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*#
_output_shapes
:���������*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*#
_output_shapes
:���������
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*#
_output_shapes
:���������
z
gradients/Neg_1_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*#
_output_shapes
:���������
e
gradients/Neg_grad/NegNeggradients/Neg_1_grad/Neg*
T0*#
_output_shapes
:���������
�
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:���������
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:���������
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Neg_grad/Negegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:���������
�
Cgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGradZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
_output_shapes
:*
T0*
data_formatNHWC
�
Hgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp[^gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulD^gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad
�
Pgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentityZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*m
_classc
a_loc:@gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0*'
_output_shapes
:���������
�
Rgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1MatMul#policy_network/fully_connected/ReluPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
Ggradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1
�
Ogradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Qgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
;gradients/policy_network/fully_connected/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency#policy_network/fully_connected/Relu*
T0*'
_output_shapes
:���������
�
Agradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
�
Fgradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/Relu_grad/ReluGradB^gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ngradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/Relu_grad/ReluGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
Pgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
;gradients/policy_network/fully_connected/MatMul_grad/MatMulMatMulNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency+policy_network/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1MatMuloNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
Egradients/policy_network/fully_connected/MatMul_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/MatMul_grad/MatMul>^gradients/policy_network/fully_connected/MatMul_grad/MatMul_1
�
Mgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/MatMul_grad/MatMulF^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Ogradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1Identity=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1F^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
beta1_power/initial_valueConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta1_power/readIdentitybeta1_power*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
beta2_power/initial_valueConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta2_power/readIdentitybeta2_power*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
=policy_network/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
�
+policy_network/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*9
_class/
-+loc:@policy_network/fully_connected/weights*
shared_name 
�
2policy_network/fully_connected/weights/Adam/AssignAssign+policy_network/fully_connected/weights/Adam=policy_network/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
�
0policy_network/fully_connected/weights/Adam/readIdentity+policy_network/fully_connected/weights/Adam*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
?policy_network/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
�
-policy_network/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*9
_class/
-+loc:@policy_network/fully_connected/weights*
shared_name 
�
4policy_network/fully_connected/weights/Adam_1/AssignAssign-policy_network/fully_connected/weights/Adam_1?policy_network/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
�
2policy_network/fully_connected/weights/Adam_1/readIdentity-policy_network/fully_connected/weights/Adam_1*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
<policy_network/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
�
*policy_network/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
1policy_network/fully_connected/biases/Adam/AssignAssign*policy_network/fully_connected/biases/Adam<policy_network/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
�
/policy_network/fully_connected/biases/Adam/readIdentity*policy_network/fully_connected/biases/Adam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
�
>policy_network/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
�
,policy_network/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
3policy_network/fully_connected/biases/Adam_1/AssignAssign,policy_network/fully_connected/biases/Adam_1>policy_network/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
�
1policy_network/fully_connected/biases/Adam_1/readIdentity,policy_network/fully_connected/biases/Adam_1*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
�
?policy_network/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
�
-policy_network/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
�
4policy_network/fully_connected_1/weights/Adam/AssignAssign-policy_network/fully_connected_1/weights/Adam?policy_network/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
�
2policy_network/fully_connected_1/weights/Adam/readIdentity-policy_network/fully_connected_1/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
�
/policy_network/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
�
6policy_network/fully_connected_1/weights/Adam_1/AssignAssign/policy_network/fully_connected_1/weights/Adam_1Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
�
4policy_network/fully_connected_1/weights/Adam_1/readIdentity/policy_network/fully_connected_1/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
>policy_network/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
�
,policy_network/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
�
3policy_network/fully_connected_1/biases/Adam/AssignAssign,policy_network/fully_connected_1/biases/Adam>policy_network/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
�
1policy_network/fully_connected_1/biases/Adam/readIdentity,policy_network/fully_connected_1/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
�
@policy_network/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
�
.policy_network/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
�
5policy_network/fully_connected_1/biases/Adam_1/AssignAssign.policy_network/fully_connected_1/biases/Adam_1@policy_network/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
�
3policy_network/fully_connected_1/biases/Adam_1/readIdentity.policy_network/fully_connected_1/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *���<*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
<Adam/update_policy_network/fully_connected/weights/ApplyAdam	ApplyAdam&policy_network/fully_connected/weights+policy_network/fully_connected/weights/Adam-policy_network/fully_connected/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonOgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking( *
T0*
_output_shapes

:
�
;Adam/update_policy_network/fully_connected/biases/ApplyAdam	ApplyAdam%policy_network/fully_connected/biases*policy_network/fully_connected/biases/Adam,policy_network/fully_connected/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonPgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking( *
T0*
_output_shapes
:
�
>Adam/update_policy_network/fully_connected_1/weights/ApplyAdam	ApplyAdam(policy_network/fully_connected_1/weights-policy_network/fully_connected_1/weights/Adam/policy_network/fully_connected_1/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:
�
=Adam/update_policy_network/fully_connected_1/biases/ApplyAdam	ApplyAdam'policy_network/fully_connected_1/biases,policy_network/fully_connected_1/biases/Adam.policy_network/fully_connected_1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/Assign^Adam/Assign_1
K

avg_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
K

max_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
K

std_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
L
eval_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
Z
Avg_Reward/tagsConst*
dtype0*
valueB B
Avg_Reward*
_output_shapes
: 
Y

Avg_RewardScalarSummaryAvg_Reward/tags
avg_reward*
T0*
_output_shapes
: 
Z
Max_Reward/tagsConst*
dtype0*
valueB B
Max_Reward*
_output_shapes
: 
Y

Max_RewardScalarSummaryMax_Reward/tags
max_reward*
T0*
_output_shapes
: 
Z
Std_Reward/tagsConst*
dtype0*
valueB B
Std_Reward*
_output_shapes
: 
Y

Std_RewardScalarSummaryStd_Reward/tags
std_reward*
T0*
_output_shapes
: 
\
Eval_Reward/tagsConst*
dtype0*
valueB BEval_Reward*
_output_shapes
: 
\
Eval_RewardScalarSummaryEval_Reward/tagseval_reward*
T0*
_output_shapes
: 
t
Merge/MergeSummaryMergeSummary
Avg_Reward
Max_Reward
Std_RewardEval_Reward*
_output_shapes
: *
N"��d�      r\�G	�g#�@��AJ׸
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.5.02v1.5.0-0-g37aa430d84��
d
oPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
a
a_discPlaceholder*
dtype0*
shape:���������*#
_output_shapes
:���������
`
advtgPlaceholder*
dtype0*
shape:���������*#
_output_shapes
:���������
�
Gpolicy_network/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB"      *
_output_shapes
:
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�7�*
_output_shapes
: 
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�7?*
_output_shapes
: 
�
Opolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformGpolicy_network/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*9
_class/
-+loc:@policy_network/fully_connected/weights
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/subSubEpolicy_network/fully_connected/weights/Initializer/random_uniform/maxEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes
: 
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/mulMulOpolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformEpolicy_network/fully_connected/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
Apolicy_network/fully_connected/weights/Initializer/random_uniformAddEpolicy_network/fully_connected/weights/Initializer/random_uniform/mulEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
&policy_network/fully_connected/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*9
_class/
-+loc:@policy_network/fully_connected/weights*
shared_name 
�
-policy_network/fully_connected/weights/AssignAssign&policy_network/fully_connected/weightsApolicy_network/fully_connected/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
�
+policy_network/fully_connected/weights/readIdentity&policy_network/fully_connected/weights*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
7policy_network/fully_connected/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
�
%policy_network/fully_connected/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
,policy_network/fully_connected/biases/AssignAssign%policy_network/fully_connected/biases7policy_network/fully_connected/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
�
*policy_network/fully_connected/biases/readIdentity%policy_network/fully_connected/biases*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
�
%policy_network/fully_connected/MatMulMatMulo+policy_network/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
&policy_network/fully_connected/BiasAddBiasAdd%policy_network/fully_connected/MatMul*policy_network/fully_connected/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
�
#policy_network/fully_connected/ReluRelu&policy_network/fully_connected/BiasAdd*
T0*'
_output_shapes
:���������
�
Ipolicy_network/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB"      *
_output_shapes
:
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *:��*
_output_shapes
: 
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *:�?*
_output_shapes
: 
�
Qpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*;
_class1
/-loc:@policy_network/fully_connected_1/weights
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/subSubGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes
: 
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulMulQpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
Cpolicy_network/fully_connected_1/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
(policy_network/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
�
/policy_network/fully_connected_1/weights/AssignAssign(policy_network/fully_connected_1/weightsCpolicy_network/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
�
-policy_network/fully_connected_1/weights/readIdentity(policy_network/fully_connected_1/weights*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
9policy_network/fully_connected_1/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
�
'policy_network/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
�
.policy_network/fully_connected_1/biases/AssignAssign'policy_network/fully_connected_1/biases9policy_network/fully_connected_1/biases/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
�
,policy_network/fully_connected_1/biases/readIdentity'policy_network/fully_connected_1/biases*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
�
'policy_network/fully_connected_1/MatMulMatMul#policy_network/fully_connected/Relu-policy_network/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
(policy_network/fully_connected_1/BiasAddBiasAdd'policy_network/fully_connected_1/MatMul,policy_network/fully_connected_1/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
e
#multinomial/Multinomial/num_samplesConst*
dtype0*
value	B :*
_output_shapes
: 
�
multinomial/MultinomialMultinomial(policy_network/fully_connected_1/BiasAdd#multinomial/Multinomial/num_samples*
seed2 *

seed *
T0*
output_dtype0	*'
_output_shapes
:���������
p
SqueezeSqueezemultinomial/Multinomial*
squeeze_dims
*
T0	*#
_output_shapes
:���������
o
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapea_disc*
out_type0*
T0*
_output_shapes
:
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits(policy_network/fully_connected_1/BiasAdda_disc*
T0*
Tlabels0*6
_output_shapes$
":���������:���������
�
NegNegGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:���������
?
Neg_1NegNeg*
T0*#
_output_shapes
:���������
F
mulMulNeg_1advtg*
T0*#
_output_shapes
:���������
R
gradients/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
j
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*#
_output_shapes
:���������
]
gradients/mul_grad/ShapeShapeNeg_1*
out_type0*
T0*
_output_shapes
:
_
gradients/mul_grad/Shape_1Shapeadvtg*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
b
gradients/mul_grad/mulMulgradients/Filladvtg*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
d
gradients/mul_grad/mul_1MulNeg_1gradients/Fill*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:���������
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*#
_output_shapes
:���������
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*#
_output_shapes
:���������
z
gradients/Neg_1_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*#
_output_shapes
:���������
e
gradients/Neg_grad/NegNeggradients/Neg_1_grad/Neg*
T0*#
_output_shapes
:���������
�
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:���������
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:���������
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Neg_grad/Negegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:���������
�
Cgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGradZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
data_formatNHWC*
T0*
_output_shapes
:
�
Hgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp[^gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulD^gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad
�
Pgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentityZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*m
_classc
a_loc:@gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0*'
_output_shapes
:���������
�
Rgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1MatMul#policy_network/fully_connected/ReluPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
Ggradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1
�
Ogradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Qgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
;gradients/policy_network/fully_connected/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency#policy_network/fully_connected/Relu*
T0*'
_output_shapes
:���������
�
Agradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
Fgradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/Relu_grad/ReluGradB^gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ngradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/Relu_grad/ReluGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
Pgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
;gradients/policy_network/fully_connected/MatMul_grad/MatMulMatMulNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency+policy_network/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1MatMuloNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
Egradients/policy_network/fully_connected/MatMul_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/MatMul_grad/MatMul>^gradients/policy_network/fully_connected/MatMul_grad/MatMul_1
�
Mgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/MatMul_grad/MatMulF^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Ogradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1Identity=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1F^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
beta1_power/initial_valueConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta1_power/readIdentitybeta1_power*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
beta2_power/initial_valueConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta2_power/readIdentitybeta2_power*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
=policy_network/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
�
+policy_network/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*9
_class/
-+loc:@policy_network/fully_connected/weights*
shared_name 
�
2policy_network/fully_connected/weights/Adam/AssignAssign+policy_network/fully_connected/weights/Adam=policy_network/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
�
0policy_network/fully_connected/weights/Adam/readIdentity+policy_network/fully_connected/weights/Adam*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
?policy_network/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
�
-policy_network/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*9
_class/
-+loc:@policy_network/fully_connected/weights*
shared_name 
�
4policy_network/fully_connected/weights/Adam_1/AssignAssign-policy_network/fully_connected/weights/Adam_1?policy_network/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
�
2policy_network/fully_connected/weights/Adam_1/readIdentity-policy_network/fully_connected/weights/Adam_1*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
�
<policy_network/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
�
*policy_network/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
1policy_network/fully_connected/biases/Adam/AssignAssign*policy_network/fully_connected/biases/Adam<policy_network/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
�
/policy_network/fully_connected/biases/Adam/readIdentity*policy_network/fully_connected/biases/Adam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
�
>policy_network/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
�
,policy_network/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@policy_network/fully_connected/biases*
shared_name 
�
3policy_network/fully_connected/biases/Adam_1/AssignAssign,policy_network/fully_connected/biases/Adam_1>policy_network/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
�
1policy_network/fully_connected/biases/Adam_1/readIdentity,policy_network/fully_connected/biases/Adam_1*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
�
?policy_network/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
�
-policy_network/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
�
4policy_network/fully_connected_1/weights/Adam/AssignAssign-policy_network/fully_connected_1/weights/Adam?policy_network/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
�
2policy_network/fully_connected_1/weights/Adam/readIdentity-policy_network/fully_connected_1/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
�
/policy_network/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
�
6policy_network/fully_connected_1/weights/Adam_1/AssignAssign/policy_network/fully_connected_1/weights/Adam_1Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
�
4policy_network/fully_connected_1/weights/Adam_1/readIdentity/policy_network/fully_connected_1/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
�
>policy_network/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
�
,policy_network/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
�
3policy_network/fully_connected_1/biases/Adam/AssignAssign,policy_network/fully_connected_1/biases/Adam>policy_network/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
�
1policy_network/fully_connected_1/biases/Adam/readIdentity,policy_network/fully_connected_1/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
�
@policy_network/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
�
.policy_network/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
�
5policy_network/fully_connected_1/biases/Adam_1/AssignAssign.policy_network/fully_connected_1/biases/Adam_1@policy_network/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
�
3policy_network/fully_connected_1/biases/Adam_1/readIdentity.policy_network/fully_connected_1/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *���<*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
<Adam/update_policy_network/fully_connected/weights/ApplyAdam	ApplyAdam&policy_network/fully_connected/weights+policy_network/fully_connected/weights/Adam-policy_network/fully_connected/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonOgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking( *
T0*
_output_shapes

:
�
;Adam/update_policy_network/fully_connected/biases/ApplyAdam	ApplyAdam%policy_network/fully_connected/biases*policy_network/fully_connected/biases/Adam,policy_network/fully_connected/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonPgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking( *
T0*
_output_shapes
:
�
>Adam/update_policy_network/fully_connected_1/weights/ApplyAdam	ApplyAdam(policy_network/fully_connected_1/weights-policy_network/fully_connected_1/weights/Adam/policy_network/fully_connected_1/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:
�
=Adam/update_policy_network/fully_connected_1/biases/ApplyAdam	ApplyAdam'policy_network/fully_connected_1/biases,policy_network/fully_connected_1/biases/Adam.policy_network/fully_connected_1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/Assign^Adam/Assign_1
K

avg_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
K

max_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
K

std_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
L
eval_rewardPlaceholder*
dtype0*
shape: *
_output_shapes
: 
Z
Avg_Reward/tagsConst*
dtype0*
valueB B
Avg_Reward*
_output_shapes
: 
Y

Avg_RewardScalarSummaryAvg_Reward/tags
avg_reward*
T0*
_output_shapes
: 
Z
Max_Reward/tagsConst*
dtype0*
valueB B
Max_Reward*
_output_shapes
: 
Y

Max_RewardScalarSummaryMax_Reward/tags
max_reward*
T0*
_output_shapes
: 
Z
Std_Reward/tagsConst*
dtype0*
valueB B
Std_Reward*
_output_shapes
: 
Y

Std_RewardScalarSummaryStd_Reward/tags
std_reward*
T0*
_output_shapes
: 
\
Eval_Reward/tagsConst*
dtype0*
valueB BEval_Reward*
_output_shapes
: 
\
Eval_RewardScalarSummaryEval_Reward/tagseval_reward*
T0*
_output_shapes
: 
t
Merge/MergeSummaryMergeSummary
Avg_Reward
Max_Reward
Std_RewardEval_Reward*
N*
_output_shapes
: ""
train_op

Adam"�
	variables��
�
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
�
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
�
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
-policy_network/fully_connected/weights/Adam:02policy_network/fully_connected/weights/Adam/Assign2policy_network/fully_connected/weights/Adam/read:02?policy_network/fully_connected/weights/Adam/Initializer/zeros:0
�
/policy_network/fully_connected/weights/Adam_1:04policy_network/fully_connected/weights/Adam_1/Assign4policy_network/fully_connected/weights/Adam_1/read:02Apolicy_network/fully_connected/weights/Adam_1/Initializer/zeros:0
�
,policy_network/fully_connected/biases/Adam:01policy_network/fully_connected/biases/Adam/Assign1policy_network/fully_connected/biases/Adam/read:02>policy_network/fully_connected/biases/Adam/Initializer/zeros:0
�
.policy_network/fully_connected/biases/Adam_1:03policy_network/fully_connected/biases/Adam_1/Assign3policy_network/fully_connected/biases/Adam_1/read:02@policy_network/fully_connected/biases/Adam_1/Initializer/zeros:0
�
/policy_network/fully_connected_1/weights/Adam:04policy_network/fully_connected_1/weights/Adam/Assign4policy_network/fully_connected_1/weights/Adam/read:02Apolicy_network/fully_connected_1/weights/Adam/Initializer/zeros:0
�
1policy_network/fully_connected_1/weights/Adam_1:06policy_network/fully_connected_1/weights/Adam_1/Assign6policy_network/fully_connected_1/weights/Adam_1/read:02Cpolicy_network/fully_connected_1/weights/Adam_1/Initializer/zeros:0
�
.policy_network/fully_connected_1/biases/Adam:03policy_network/fully_connected_1/biases/Adam/Assign3policy_network/fully_connected_1/biases/Adam/read:02@policy_network/fully_connected_1/biases/Adam/Initializer/zeros:0
�
0policy_network/fully_connected_1/biases/Adam_1:05policy_network/fully_connected_1/biases/Adam_1/Assign5policy_network/fully_connected_1/biases/Adam_1/read:02Bpolicy_network/fully_connected_1/biases/Adam_1/Initializer/zeros:0"H
	summaries;
9
Avg_Reward:0
Max_Reward:0
Std_Reward:0
Eval_Reward:0"�
model_variables��
�
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
�
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
�
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0"�
trainable_variables��
�
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
�
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
�
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0�>kaX       ��/�	�ϐ@��A*M


Avg_Reward��A


Max_Reward  @B


Std_Reward|l�?

Eval_Reward  0A�ǎZ       o��	����@��A*M


Avg_RewardW��A


Max_Reward  �B


Std_Reward�*�?

Eval_Reward  BBn-�Z       o��	[��@��A*M


Avg_RewardDDB


Max_Reward  �B


Std_Reward�U7@

Eval_Reward  pB8�ghZ       o��	pw�@��A*M


Avg_Rewardz�+B


Max_Reward  �B


Std_Reward���@

Eval_Reward  �AC��dZ       o��	��@��A*M


Avg_RewardUU"B


Max_Reward  �B


Std_RewardvY�@

Eval_Reward  �A��@Z       o��	d�)�@��A*M


Avg_RewardnۇB


Max_Reward  �B


Std_Reward��A

Eval_Reward  �B_BRZ       o��	��:�@��A*M


Avg_Reward�E�B


Max_Reward  HC


Std_Reward�GgA

Eval_Reward  �B�uO�Z       o��	[BK�@��A*M


Avg_Reward  �B


Max_Reward  >C


Std_Reward\-A

Eval_Reward  �BBC�Z       o��	#�[�@��A*M


Avg_Reward�̹B


Max_Reward  HC


Std_Reward�^A

Eval_Reward  hB���Z       o��	Eel�@��A	*M


Avg_Reward  �B


Max_Reward  HC


Std_Rewardt�NA

Eval_Reward  �B��Z       o��	��|�@��A
*M


Avg_RewardUUC


Max_Reward  HC


Std_Reward5<MA

Eval_Reward  C���Z       o��	e��@��A*M


Avg_Reward  C


Max_Reward  HC


Std_Reward�ŝA

Eval_Reward  C!��Z       o��	����@��A*M


Avg_Reward�* C


Max_Reward  HC


Std_Reward2�]A

Eval_Reward  �BxLZ       o��	�J��@��A*M


Avg_Reward33@C


Max_Reward  HC


Std_Reward�?�@

Eval_Reward  HC��UdZ       o��	����@��A*M


Avg_Rewardff/C


Max_Reward  HC


Std_Reward��WA

Eval_Reward  	C���/Z       o��	�/ё@��A*M


Avg_Reward331C


Max_Reward  HC


Std_RewardM�IA

Eval_Reward  	C���Z       o��	3��@��A*M


Avg_Reward  2C


Max_Reward  HC


Std_Reward��fA

Eval_Reward  HC�I+Z       o��	���@��A*M


Avg_Reward��FC


Max_Reward  HC


Std_RewardH�?

Eval_Reward  HC���Z       o��		�@��A*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCn�7PZ       o��	���@��A*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC8m?�Z       o��	&�(�@��A*M


Avg_RewardU�C


Max_Reward  HC


Std_Reward"f�A

Eval_Reward  �B���CZ       o��	��9�@��A*M


Avg_Reward��,C


Max_Reward  HC


Std_Reward�{�A

Eval_Reward  HC���%Z       o��	��J�@��A*M


Avg_Rewardff.C


Max_Reward  HC


Std_Reward�dA

Eval_Reward  �BH~:�Z       o��	I\�@��A*M


Avg_Reward  5C


Max_Reward  HC


Std_Reward��A

Eval_Reward  HC��٨Z       o��	��l�@��A*M


Avg_Rewardff5C


Max_Reward  HC


Std_RewardQt1A

Eval_Reward  HC0=a�Z       o��	�n}�@��A*M


Avg_Rewardff5C


Max_Reward  HC


Std_Reward<�A

Eval_Reward  HC��Z       o��	����@��A*M


Avg_Reward��?C


Max_Reward  HC


Std_Rewardzt@

Eval_Reward  DC~��Z       o��	m��@��A*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC�m�xZ       o��	����@��A*M


Avg_Reward��-C


Max_Reward  HC


Std_Reward,�A

Eval_Reward  �B�<�Z       o��	��@��A*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC*��yZ       o��	��Ӓ@��A*M


Avg_RewardffCC


Max_Reward  HC


Std_Rewardᨃ@

Eval_Reward  HC"yǵZ       o��	�P�@��A*M


Avg_Reward  3C


Max_Reward  HC


Std_Reward�C�A

Eval_Reward  HCL.�Z       o��	���@��A *M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��Z       o��	�w�@��A!*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��r�Z       o��	T�@��A"*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCDX�Z       o��	�l*�@��A#*M


Avg_Reward��-C


Max_Reward  HC


Std_Reward,�A

Eval_Reward  HC�īrZ       o��	��:�@��A$*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC`��wZ       o��	�kK�@��A%*M


Avg_Reward��7C


Max_Reward  HC


Std_Reward��gA

Eval_Reward  �B����Z       o��	�)\�@��A&*M


Avg_Reward��EC


Max_Reward  HC


Std_RewardOb	@

Eval_Reward  HC��Z       o��	�l�@��A'*M


Avg_RewardUU"C


Max_Reward  HC


Std_Rewardcw|A

Eval_Reward  �B&�i�Z       o��	k�|�@��A(*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC*�Z       o��	�U��@��A)*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC}ȕxZ       o��	���@��A**M


Avg_Reward  GC


Max_Reward  HC


Std_Reward.�d?

Eval_Reward  CCv5�Z       o��	A���@��A+*M


Avg_Reward��>C


Max_Reward  HC


Std_Reward�A

Eval_Reward  HC��d�Z       o��	�J��@��A,*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC�Z       o��	��ѓ@��A-*M


Avg_Reward  ;C


Max_Reward  HC


Std_Rewarde�@

Eval_Reward  HC�I�Z       o��	��@��A.*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC3̔�Z       o��	���@��A/*M


Avg_Reward33AC


Max_Reward  HC


Std_Reward���@

Eval_Reward  &C`���Z       o��	2r�@��A0*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC~��Z       o��	��@��A1*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCJ[�Z       o��	�%�@��A2*M


Avg_Reward33BC


Max_Reward  HC


Std_Rewardu�@

Eval_Reward  HC�U��Z       o��	g�6�@��A3*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��QZ       o��	�9G�@��A4*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCwQ(�Z       o��	��W�@��A5*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��Z       o��	��h�@��A6*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC	}	�Z       o��	U�y�@��A7*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��Z       o��	�U��@��A8*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��ZjZ       o��	�֜�@��A9*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCtZnZ       o��	Ɖ��@��A:*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��/)Z       o��	����@��A;*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC��a�Z       o��	��є@��A<*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC%�0Z       o��	��@��A=*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC"�Z�Z       o��	���@��A>*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC}�Z       o��	���@��A?*M


Avg_Reward��@C


Max_Reward  HC


Std_Rewardv�@

Eval_Reward  HC���lZ       o��	��@��A@*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC� �!Z       o��	u&�@��AA*M


Avg_Rewardff'C


Max_Reward  HC


Std_RewardAD�A

Eval_Reward  B���Z       o��	�59�@��AB*M


Avg_Reward33DC


Max_Reward  HC


Std_RewardR�Y@

Eval_Reward  HC�-ݤZ       o��	�J�@��AC*M


Avg_Reward�*C


Max_Reward  HC


Std_Reward��A

Eval_Reward  �B	���Z       o��	q\�@��AD*M


Avg_Reward��C


Max_Reward  HC


Std_Reward� B

Eval_Reward  HC��Z       o��	W�l�@��AE*M


Avg_Reward��!C


Max_Reward  HC


Std_Reward�XB

Eval_Reward  �A��.cZ       o��	sL}�@��AF*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC<��Z       o��	�Ӎ�@��AG*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC܇oZ       o��	Vd��@��AH*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC�Q'Z       o��	�߯�@��AI*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCŀ��Z       o��	�J@��AJ*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC5�[�Z       o��	��ҕ@��AK*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HCl�aZ       o��	�^�@��AL*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC{�JZ       o��	���@��AM*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC=�eZ       o��	L7�@��AN*M


Avg_Reward  HC


Max_Reward  HC


Std_Reward    

Eval_Reward  HC�7	�Z       o��	���@��AO*M


Avg_Reward��EC


Max_Reward  HC


Std_RewardOb	@

Eval_Reward  HC��Z       o��	�P%�@��AP*M


Avg_Reward��.C


Max_Reward  9C


Std_RewardT�~@

Eval_Reward  !C�beLZ       o��	��6�@��AQ*M


Avg_Rewardff,C


Max_Reward  7C


Std_Reward�rP@

Eval_Reward  0C��mZ       o��	�H�@��AR*M


Avg_Reward��'C


Max_Reward  2C


Std_Reward���@

Eval_Reward  2C�z�|Z       o��	!Y�@��AS*M


Avg_Reward �C


Max_Reward  C


Std_RewardW�@

Eval_Reward  C~=�Z       o��	�wi�@��AT*M


Avg_Reward�*C


Max_Reward  (C


Std_Rewardz�S@

Eval_Reward  (C�l-Z       o��	�y�@��AU*M


Avg_Reward�*C


Max_Reward  )C


Std_Reward�@

Eval_Reward  
C5;HZ       o��	�o��@��AV*M


Avg_RewardUUC


Max_Reward  (C


Std_RewardǪ@

Eval_Reward  CkV��Z       o��	ǟ��@��AW*M


Avg_Reward��C


Max_Reward  C


Std_RewardO5?@

Eval_Reward  CIL�Z       o��	B$��@��AX*M


Avg_Reward �C


Max_Reward  "C


Std_Reward`�N@

Eval_Reward   Cej�Z       o��	76��@��AY*M


Avg_Reward��C


Max_Reward   C


Std_Reward�͊@

Eval_Reward   Ca	Z       o��	s,ϖ@��AZ*M


Avg_Reward�mC


Max_Reward  C


Std_Reward�h@

Eval_Reward  C��_Z       o��	^��@��A[*M


Avg_Reward�$C


Max_Reward  C


Std_RewardE}�@

Eval_Reward  C��_RZ       o��	��@��A\*M


Avg_Reward  �B


Max_Reward  #C


Std_Reward+<�@

Eval_Reward  �B#R0>Z       o��	75�@��A]*M


Avg_Reward  �B


Max_Reward  �B


Std_Reward8�9@

Eval_Reward  �B�G�Z       o��	i��@��A^*M


Avg_Reward @�B


Max_Reward  C


Std_Reward��d@

Eval_Reward  �B�do?Z       o��	%]%�@��A_*M


Avg_Reward ��B


Max_Reward  C


Std_Reward�V�@

Eval_Reward  �BY�{�Z       o��	��5�@��A`*M


Avg_Reward ��B


Max_Reward   C


Std_Reward�=>@

Eval_Reward  �B�,�Z       o��	� I�@��Aa*M


Avg_Reward  �B


Max_Reward  C


Std_RewardU��@

Eval_Reward  �B��,Z       o��	�Z�@��Ab*M


Avg_Reward ��B


Max_Reward  C


Std_Reward��@

Eval_Reward  C�b�Z       o��	��j�@��Ac*M


Avg_Reward  �B


Max_Reward  C


Std_Reward���@

Eval_Reward  �BD&(