       �K"	  ��Ң�Abrain.Event:2):���     ��A
	�ؖҢ�A"��
d
oPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
{
a_contPlaceholder*
dtype0*%
shape:������������������*0
_output_shapes
:������������������
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
valueB"       *
_output_shapes
:
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�)��*
_output_shapes
: 
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�)�>*
_output_shapes
: 
�
Opolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformGpolicy_network/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

: *
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

: 
�
Apolicy_network/fully_connected/weights/Initializer/random_uniformAddEpolicy_network/fully_connected/weights/Initializer/random_uniform/mulEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
&policy_network/fully_connected/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *9
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

: 
�
+policy_network/fully_connected/weights/readIdentity&policy_network/fully_connected/weights*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
7policy_network/fully_connected/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
%policy_network/fully_connected/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
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
: 
�
*policy_network/fully_connected/biases/readIdentity%policy_network/fully_connected/biases*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
%policy_network/fully_connected/MatMulMatMulo+policy_network/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
&policy_network/fully_connected/BiasAddBiasAdd%policy_network/fully_connected/MatMul*policy_network/fully_connected/biases/read*'
_output_shapes
:��������� *
T0*
data_formatNHWC
�
#policy_network/fully_connected/ReluRelu&policy_network/fully_connected/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Ipolicy_network/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB"        *
_output_shapes
:
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *qĜ�*
_output_shapes
: 
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *qĜ>*
_output_shapes
: 
�
Qpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:  *
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

:  
�
Cpolicy_network/fully_connected_1/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
(policy_network/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *;
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

:  
�
-policy_network/fully_connected_1/weights/readIdentity(policy_network/fully_connected_1/weights*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
9policy_network/fully_connected_1/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
'policy_network/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *:
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
: 
�
,policy_network/fully_connected_1/biases/readIdentity'policy_network/fully_connected_1/biases*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
: 
�
'policy_network/fully_connected_1/MatMulMatMul#policy_network/fully_connected/Relu-policy_network/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
(policy_network/fully_connected_1/BiasAddBiasAdd'policy_network/fully_connected_1/MatMul,policy_network/fully_connected_1/biases/read*'
_output_shapes
:��������� *
T0*
data_formatNHWC
�
%policy_network/fully_connected_1/ReluRelu(policy_network/fully_connected_1/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Ipolicy_network/fully_connected_2/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB"       *
_output_shapes
:
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB
 *�r˾*
_output_shapes
: 
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB
 *�r�>*
_output_shapes
: 
�
Qpolicy_network/fully_connected_2/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_2/weights/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*;
_class1
/-loc:@policy_network/fully_connected_2/weights
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/subSubGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/maxGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes
: 
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/mulMulQpolicy_network/fully_connected_2/weights/Initializer/random_uniform/RandomUniformGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
Cpolicy_network/fully_connected_2/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
(policy_network/fully_connected_2/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
shared_name 
�
/policy_network/fully_connected_2/weights/AssignAssign(policy_network/fully_connected_2/weightsCpolicy_network/fully_connected_2/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
-policy_network/fully_connected_2/weights/readIdentity(policy_network/fully_connected_2/weights*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
9policy_network/fully_connected_2/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
'policy_network/fully_connected_2/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
shared_name 
�
.policy_network/fully_connected_2/biases/AssignAssign'policy_network/fully_connected_2/biases9policy_network/fully_connected_2/biases/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
,policy_network/fully_connected_2/biases/readIdentity'policy_network/fully_connected_2/biases*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
T0*
_output_shapes
:
�
'policy_network/fully_connected_2/MatMulMatMul%policy_network/fully_connected_1/Relu-policy_network/fully_connected_2/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
(policy_network/fully_connected_2/BiasAddBiasAdd'policy_network/fully_connected_2/MatMul,policy_network/fully_connected_2/biases/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
�
(log_std/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@log_std*
valueB:*
_output_shapes
:
�
&log_std/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@log_std*
valueB
 *�5�*
_output_shapes
: 
�
&log_std/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@log_std*
valueB
 *�5?*
_output_shapes
: 
�
0log_std/Initializer/random_uniform/RandomUniformRandomUniform(log_std/Initializer/random_uniform/shape*
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*
_class
loc:@log_std
�
&log_std/Initializer/random_uniform/subSub&log_std/Initializer/random_uniform/max&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
&log_std/Initializer/random_uniform/mulMul0log_std/Initializer/random_uniform/RandomUniform&log_std/Initializer/random_uniform/sub*
_class
loc:@log_std*
T0*
_output_shapes
:
�
"log_std/Initializer/random_uniformAdd&log_std/Initializer/random_uniform/mul&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
:
�
log_std
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
�
log_std/AssignAssignlog_std"log_std/Initializer/random_uniform*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
b
log_std/readIdentitylog_std*
_class
loc:@log_std*
T0*
_output_shapes
:
=
ExpExplog_std/read*
T0*
_output_shapes
:
d
random_normal/shapeConst*
dtype0*
valueB"�a     *
_output_shapes
:
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
l
random_normal/mulMul"random_normal/RandomStandardNormalExp*
T0* 
_output_shapes
:
��
|
random_normalAddrandom_normal/mul(policy_network/fully_connected_2/BiasAdd*
T0* 
_output_shapes
:
��
?
Exp_1Explog_std/read*
T0*
_output_shapes
:
�
bMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
�
fMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag_1/batch_shape_tensor/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
8MultivariateNormalDiag_1/shapes_from_loc_and_scale/ShapeShape(policy_network/fully_connected_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
FMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_1Const*
dtype0*
valueB:
���������*
_output_shapes
:
�
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
@MultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_sliceStridedSlice8MultivariateNormalDiag_1/shapes_from_loc_and_scale/ShapeFMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stackHMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_1HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
^MultivariateNormalDiag_1/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsfMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag_1/batch_shape_tensor/batch_shape@MultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice*
T0*
_output_shapes
:
J
zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
I
onesConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
>

Normal/locIdentityzeros*
T0*
_output_shapes
: 
?
Normal/scaleIdentityones*
T0*
_output_shapes
: 
i
'affine_linear_operator/init/event_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
>affine_linear_operator/init/DistributionShape/init/batch_ndimsConst*
dtype0*
value	B : *
_output_shapes
: 
�
>affine_linear_operator/init/DistributionShape/init/event_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
c
!affine_linear_operator/init/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
_
MultivariateNormalDiag_2/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
a
MultivariateNormalDiag_2/emptyConst*
dtype0*
valueB *
_output_shapes
: 
�
?MultivariateNormalDiag_2/Normal/is_scalar_batch/is_scalar_batchConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
`
MultivariateNormalDiag_2/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_1Const*
dtype0*
value	B : *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_2Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_3Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_4Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_5Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_6Const*
dtype0*
value	B :*
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_7Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_8Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
AMultivariateNormalDiag_2/Normal_1/is_scalar_event/is_scalar_eventConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
n
$MultivariateNormalDiag_2/event_shapeConst*
dtype0*
valueB:*
_output_shapes
:
b
 MultivariateNormalDiag_2/Const_9Const*
dtype0*
value	B :*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_10Const*
dtype0*
value	B : *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_11Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_12Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_13Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_14Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_15Const*
dtype0*
value	B :*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_16Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_17Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_18Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
AMultivariateNormalDiag_2/Normal_2/is_scalar_batch/is_scalar_batchConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_19Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_20Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_21Const*
dtype0*
value	B :*
_output_shapes
: 
`
MultivariateNormalDiag_2/sub/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
MultivariateNormalDiag_2/subSubMultivariateNormalDiag_2/sub/x!MultivariateNormalDiag_2/Const_21*
T0*
_output_shapes
: 
f
$MultivariateNormalDiag_2/range/limitConst*
dtype0*
value	B : *
_output_shapes
: 
f
$MultivariateNormalDiag_2/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
MultivariateNormalDiag_2/rangeRangeMultivariateNormalDiag_2/sub$MultivariateNormalDiag_2/range/limit$MultivariateNormalDiag_2/range/delta*

Tidx0*#
_output_shapes
:���������
�
DMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/subSuba_cont(policy_network/fully_connected_2/BiasAdd*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAdd�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/x�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
N*
T0*
_output_shapes
:*

axis 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
N*
T0*
_output_shapes
:*

axis 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSlice�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/begin�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/pick_vector/condConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concatConcatV2�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/ReshapeReshapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat*'
_output_shapes
:���������*
T0*
Tshape0
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*'
_output_shapes
:���������
�
cMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
aMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivRealDivcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xExp_1*
T0*
_output_shapes
:
�
hMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
dMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims
ExpandDimsaMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivhMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mulMul�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAdd�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/x�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
N*
T0*
_output_shapes
:*

axis 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
N*
T0*
_output_shapes
:*

axis 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSlice�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/begin�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concatConcatV2�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/ReshapeReshape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat*'
_output_shapes
:���������*
T0*
Tshape0
�
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsAbsExp_1*
T0*
_output_shapes
:
�
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogLogvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/SumSumvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log�MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
WMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/NegNegvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum*
T0*
_output_shapes
: 
�
AMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subSub�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape
Normal/loc*
T0*'
_output_shapes
:���������
�
EMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truedivRealDivAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subNormal/scale*
T0*'
_output_shapes
:���������
�
8MultivariateNormalDiag_3/log_prob/Normal/log_prob/SquareSquareEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/xConst*
dtype0*
valueB
 *   �*
_output_shapes
: 
�
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mulMul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*'
_output_shapes
:���������
k
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/LogLogNormal/scale*
T0*
_output_shapes
: 
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/xConst*
dtype0*
valueB
 *�?k?*
_output_shapes
: 
�
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/addAdd7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/x5MultivariateNormalDiag_3/log_prob/Normal/log_prob/Log*
T0*
_output_shapes
: 
�
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subSub5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul5MultivariateNormalDiag_3/log_prob/Normal/log_prob/add*
T0*'
_output_shapes
:���������
�
%MultivariateNormalDiag_3/log_prob/SumSum5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subMultivariateNormalDiag_2/range*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
%MultivariateNormalDiag_3/log_prob/addAdd%MultivariateNormalDiag_3/log_prob/SumWMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg*
T0*#
_output_shapes
:���������
_
NegNeg%MultivariateNormalDiag_3/log_prob/add*
T0*#
_output_shapes
:���������
D
mulMulNegadvtg*
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
[
gradients/mul_grad/ShapeShapeNeg*
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
b
gradients/mul_grad/mul_1MulNeggradients/Fill*
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
x
gradients/Neg_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*#
_output_shapes
:���������
�
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/ShapeShape%MultivariateNormalDiag_3/log_prob/Sum*
out_type0*
T0*#
_output_shapes
:���������

<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Jgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/MultivariateNormalDiag_3/log_prob/add_grad/SumSumgradients/Neg_grad/NegJgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeReshape8gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1Sumgradients/Neg_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1Reshape:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Egradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_depsNoOp=^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape?^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1
�
Mgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyIdentity<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeF^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*O
_classE
CAloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape*
T0*
_output_shapes
:
�
Ogradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1Identity>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1F^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub*
out_type0*
T0*
_output_shapes
:
�
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/SizeConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/addAddMultivariateNormalDiag_2/range9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/modFloorMod8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/add9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1Shape8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod*
out_type0*
T0*
_output_shapes
:*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape
�
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/startConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B : *
_output_shapes
: 
�
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/deltaConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/rangeRange@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/start9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/delta*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/valueConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/FillFill<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/value*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
Bgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitchDynamicStitch:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������*
N
�
>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/yConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/MaximumMaximumBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/y*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordivFloorDiv:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*
_output_shapes
:
�
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ReshapeReshapeMgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileTile<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Reshape=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
jgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/NegNegOgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul*
out_type0*
T0*
_output_shapes
:
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumSum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1Sum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Tile\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegNegJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1ReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1
�
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1*
T0*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/SizeConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/addAdd�MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/modFloorMod�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/add�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1Const*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
valueB:*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/startConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B : *
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/deltaConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/rangeRange�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/start�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/delta*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/valueConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/FillFill�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/value*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitchDynamicStitch�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/mod�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:���������*
N
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/yConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/MaximumMaximum�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitch�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/y*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordivFloorDiv�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ReshapeReshapejgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/Neg�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/TileTile�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Reshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1Shape8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
out_type0*
T0*
_output_shapes
:
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulMul]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*'
_output_shapes
:���������
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumSumHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1Mul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1ReshapeJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1
�
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape*
T0*
_output_shapes
: 
�
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal
ReciprocalvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs�^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Tile*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mulMul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Tile�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal*
T0*
_output_shapes
:
�
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xConst`^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
�
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Mul_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/SignSignExp_1*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mulMul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/Sign*
T0*
_output_shapes
:
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ShapeShapeAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
out_type0*
T0*
_output_shapes
:
�
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivRealDivMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Normal/scale*
T0*'
_output_shapes
:���������
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumSum\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivjgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ReshapeReshapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNegAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1RealDivXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNormal/scale*
T0*'
_output_shapes
:���������
�
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2RealDiv^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1Normal/scale*
T0*'
_output_shapes
:���������
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1SumXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mullgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1ReshapeZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
egradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_depsNoOp]^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1
�
mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentity\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshapef^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*o
_classe
caloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape*
T0*'
_output_shapes
:���������
�
ogradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependency_1Identity^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1f^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*q
_classg
ecloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
�
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape*
out_type0*
T0*
_output_shapes
:
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
fgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumSummgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyfgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ReshapeReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1Summgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyhgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegNegVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1ReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
agradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_depsNoOpY^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape[^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1
�
igradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependencyIdentityXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshapeb^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*k
_classa
_]loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1b^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1*
T0*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshapeigradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependency�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutation�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	Transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Reshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*'
_output_shapes
:���������
�
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgsBroadcastGradientArgsrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mulMul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*'
_output_shapes
:���������
�
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/SumSumpgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ReshapeReshapepgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1Mul�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose*
T0*'
_output_shapes
:���������
�
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1Reshapergradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
�
}gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_depsNoOpu^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshapew^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependencyIdentitytgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*�
_class}
{yloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1Identityvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*�
_class
}{loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1*
T0*
_output_shapes

:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutation�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	Transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*'
_output_shapes
:���������
�
ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeReshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shapexgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDivRealDiv{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeExp_1*
T0*
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/SumSumxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ReshapeReshapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sumvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegNegcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/x*
T0*
_output_shapes
: 
�
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1RealDivtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegExp_1*
T0*
_output_shapes
:
�
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2RealDivzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1Exp_1*
T0*
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mulMul{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Reshapezgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2*
T0*
_output_shapes
:
�
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1Sumtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1Reshapevgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_depsNoOpy^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape{^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependencyIdentityxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape�^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*�
_class�
}loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape*
T0*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1Identityzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1�^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*�
_class�
�loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1*
T0*
_output_shapes
:
�
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ShapeShapea_cont*
out_type0*
T0*
_output_shapes
:
�
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1Shape(policy_network/fully_connected_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
igradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgsBroadcastGradientArgsYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumSum�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapeigradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ReshapeReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape*0
_output_shapes
:������������������*
T0*
Tshape0
�
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1Sum�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapekgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/NegNegYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1*
T0*
_output_shapes
:
�
]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1ReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Neg[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
dgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_depsNoOp\^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape^^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1
�
lgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependencyIdentity[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshapee^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*n
_classd
b`loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape*
T0*0
_output_shapes
:������������������
�
ngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1Identity]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1e^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
gradients/AddNAddN�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mul*
T0*
_output_shapes
:*
N
[
gradients/Exp_1_grad/mulMulgradients/AddNExp_1*
T0*
_output_shapes
:
�
Cgradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGradBiasAddGradngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*
data_formatNHWC
�
Hgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/group_depsNoOpo^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1D^gradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGrad
�
Pgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependencyIdentityngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1I^gradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Rgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
=gradients/policy_network/fully_connected_2/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_2/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
?gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1MatMul%policy_network/fully_connected_1/ReluPgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Ggradients/policy_network/fully_connected_2/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_2/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1
�
Ogradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_2/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_2/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Qgradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_2/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
=gradients/policy_network/fully_connected_1/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependency%policy_network/fully_connected_1/Relu*
T0*'
_output_shapes
:��������� 
�
Cgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad=gradients/policy_network/fully_connected_1/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
�
Hgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/Relu_grad/ReluGradD^gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad
�
Pgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/Relu_grad/ReluGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Rgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1MatMul#policy_network/fully_connected/ReluPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:  
�
Ggradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1
�
Ogradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Qgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
�
;gradients/policy_network/fully_connected/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency#policy_network/fully_connected/Relu*
T0*'
_output_shapes
:��������� 
�
Agradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
_output_shapes
: *
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
:��������� 
�
Pgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
;gradients/policy_network/fully_connected/MatMul_grad/MatMulMatMulNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency+policy_network/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1MatMuloNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Egradients/policy_network/fully_connected/MatMul_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/MatMul_grad/MatMul>^gradients/policy_network/fully_connected/MatMul_grad/MatMul_1
�
Mgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/MatMul_grad/MatMulF^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Ogradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1Identity=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1F^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
z
beta1_power/initial_valueConst*
dtype0*
_class
loc:@log_std*
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
shape: *
_class
loc:@log_std*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
: 
f
beta1_power/readIdentitybeta1_power*
_class
loc:@log_std*
T0*
_output_shapes
: 
z
beta2_power/initial_valueConst*
dtype0*
_class
loc:@log_std*
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
shape: *
_class
loc:@log_std*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
: 
f
beta2_power/readIdentitybeta2_power*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
=policy_network/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
+policy_network/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *9
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

: 
�
0policy_network/fully_connected/weights/Adam/readIdentity+policy_network/fully_connected/weights/Adam*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
?policy_network/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
-policy_network/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *9
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

: 
�
2policy_network/fully_connected/weights/Adam_1/readIdentity-policy_network/fully_connected/weights/Adam_1*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
<policy_network/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
*policy_network/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
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
: 
�
/policy_network/fully_connected/biases/Adam/readIdentity*policy_network/fully_connected/biases/Adam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
>policy_network/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
,policy_network/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
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
: 
�
1policy_network/fully_connected/biases/Adam_1/readIdentity,policy_network/fully_connected/biases/Adam_1*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
?policy_network/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
-policy_network/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *;
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

:  
�
2policy_network/fully_connected_1/weights/Adam/readIdentity-policy_network/fully_connected_1/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
/policy_network/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *;
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

:  
�
4policy_network/fully_connected_1/weights/Adam_1/readIdentity/policy_network/fully_connected_1/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
>policy_network/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
,policy_network/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *:
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
: 
�
1policy_network/fully_connected_1/biases/Adam/readIdentity,policy_network/fully_connected_1/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
: 
�
@policy_network/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
.policy_network/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *:
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
: 
�
3policy_network/fully_connected_1/biases/Adam_1/readIdentity.policy_network/fully_connected_1/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
: 
�
?policy_network/fully_connected_2/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
-policy_network/fully_connected_2/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
shared_name 
�
4policy_network/fully_connected_2/weights/Adam/AssignAssign-policy_network/fully_connected_2/weights/Adam?policy_network/fully_connected_2/weights/Adam/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
2policy_network/fully_connected_2/weights/Adam/readIdentity-policy_network/fully_connected_2/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
Apolicy_network/fully_connected_2/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
/policy_network/fully_connected_2/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
shared_name 
�
6policy_network/fully_connected_2/weights/Adam_1/AssignAssign/policy_network/fully_connected_2/weights/Adam_1Apolicy_network/fully_connected_2/weights/Adam_1/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
4policy_network/fully_connected_2/weights/Adam_1/readIdentity/policy_network/fully_connected_2/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
>policy_network/fully_connected_2/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
,policy_network/fully_connected_2/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
shared_name 
�
3policy_network/fully_connected_2/biases/Adam/AssignAssign,policy_network/fully_connected_2/biases/Adam>policy_network/fully_connected_2/biases/Adam/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
1policy_network/fully_connected_2/biases/Adam/readIdentity,policy_network/fully_connected_2/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
T0*
_output_shapes
:
�
@policy_network/fully_connected_2/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
.policy_network/fully_connected_2/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
shared_name 
�
5policy_network/fully_connected_2/biases/Adam_1/AssignAssign.policy_network/fully_connected_2/biases/Adam_1@policy_network/fully_connected_2/biases/Adam_1/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
3policy_network/fully_connected_2/biases/Adam_1/readIdentity.policy_network/fully_connected_2/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
T0*
_output_shapes
:
�
log_std/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
�
log_std/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
�
log_std/Adam/AssignAssignlog_std/Adamlog_std/Adam/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
l
log_std/Adam/readIdentitylog_std/Adam*
_class
loc:@log_std*
T0*
_output_shapes
:
�
 log_std/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
�
log_std/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
�
log_std/Adam_1/AssignAssignlog_std/Adam_1 log_std/Adam_1/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
p
log_std/Adam_1/readIdentitylog_std/Adam_1*
_class
loc:@log_std*
T0*
_output_shapes
:
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

: 
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
: 
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

:  
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
: 
�
>Adam/update_policy_network/fully_connected_2/weights/ApplyAdam	ApplyAdam(policy_network/fully_connected_2/weights-policy_network/fully_connected_2/weights/Adam/policy_network/fully_connected_2/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking( *
T0*
_output_shapes

: 
�
=Adam/update_policy_network/fully_connected_2/biases/ApplyAdam	ApplyAdam'policy_network/fully_connected_2/biases,policy_network/fully_connected_2/biases/Adam.policy_network/fully_connected_2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking( *
T0*
_output_shapes
:
�
Adam/update_log_std/ApplyAdam	ApplyAdamlog_stdlog_std/Adamlog_std/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/Exp_1_grad/mul*
use_nesterov( *
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_2/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_2/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_2/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_2/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_2/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_2/biases/ApplyAdam^Adam/update_log_std/ApplyAdam^Adam/Assign^Adam/Assign_1
�
Abaseline/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB"       *
_output_shapes
:
�
?baseline/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *�)��*
_output_shapes
: 
�
?baseline/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *�)�>*
_output_shapes
: 
�
Ibaseline/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformAbaseline/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@baseline/fully_connected/weights
�
?baseline/fully_connected/weights/Initializer/random_uniform/subSub?baseline/fully_connected/weights/Initializer/random_uniform/max?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes
: 
�
?baseline/fully_connected/weights/Initializer/random_uniform/mulMulIbaseline/fully_connected/weights/Initializer/random_uniform/RandomUniform?baseline/fully_connected/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
;baseline/fully_connected/weights/Initializer/random_uniformAdd?baseline/fully_connected/weights/Initializer/random_uniform/mul?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
 baseline/fully_connected/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
�
'baseline/fully_connected/weights/AssignAssign baseline/fully_connected/weights;baseline/fully_connected/weights/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

: 
�
%baseline/fully_connected/weights/readIdentity baseline/fully_connected/weights*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
1baseline/fully_connected/biases/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
baseline/fully_connected/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
&baseline/fully_connected/biases/AssignAssignbaseline/fully_connected/biases1baseline/fully_connected/biases/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
$baseline/fully_connected/biases/readIdentitybaseline/fully_connected/biases*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
baseline/fully_connected/MatMulMatMulo%baseline/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
 baseline/fully_connected/BiasAddBiasAddbaseline/fully_connected/MatMul$baseline/fully_connected/biases/read*'
_output_shapes
:��������� *
T0*
data_formatNHWC
y
baseline/fully_connected/ReluRelu baseline/fully_connected/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Cbaseline/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB"        *
_output_shapes
:
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *qĜ�*
_output_shapes
: 
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *qĜ>*
_output_shapes
: 
�
Kbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformCbaseline/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:  *
dtype0*
seed2 *

seed *
T0*5
_class+
)'loc:@baseline/fully_connected_1/weights
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/subSubAbaseline/fully_connected_1/weights/Initializer/random_uniform/maxAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes
: 
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/mulMulKbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformAbaseline/fully_connected_1/weights/Initializer/random_uniform/sub*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
=baseline/fully_connected_1/weights/Initializer/random_uniformAddAbaseline/fully_connected_1/weights/Initializer/random_uniform/mulAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
"baseline/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
�
)baseline/fully_connected_1/weights/AssignAssign"baseline/fully_connected_1/weights=baseline/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:  
�
'baseline/fully_connected_1/weights/readIdentity"baseline/fully_connected_1/weights*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
3baseline/fully_connected_1/biases/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
!baseline/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
�
(baseline/fully_connected_1/biases/AssignAssign!baseline/fully_connected_1/biases3baseline/fully_connected_1/biases/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
: 
�
&baseline/fully_connected_1/biases/readIdentity!baseline/fully_connected_1/biases*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
: 
�
!baseline/fully_connected_1/MatMulMatMulbaseline/fully_connected/Relu'baseline/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
"baseline/fully_connected_1/BiasAddBiasAdd!baseline/fully_connected_1/MatMul&baseline/fully_connected_1/biases/read*'
_output_shapes
:��������� *
T0*
data_formatNHWC
}
baseline/fully_connected_1/ReluRelu"baseline/fully_connected_1/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Cbaseline/fully_connected_2/weights/Initializer/random_uniform/shapeConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB"       *
_output_shapes
:
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB
 *JQھ*
_output_shapes
: 
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/maxConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB
 *JQ�>*
_output_shapes
: 
�
Kbaseline/fully_connected_2/weights/Initializer/random_uniform/RandomUniformRandomUniformCbaseline/fully_connected_2/weights/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*5
_class+
)'loc:@baseline/fully_connected_2/weights
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/subSubAbaseline/fully_connected_2/weights/Initializer/random_uniform/maxAbaseline/fully_connected_2/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes
: 
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/mulMulKbaseline/fully_connected_2/weights/Initializer/random_uniform/RandomUniformAbaseline/fully_connected_2/weights/Initializer/random_uniform/sub*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
=baseline/fully_connected_2/weights/Initializer/random_uniformAddAbaseline/fully_connected_2/weights/Initializer/random_uniform/mulAbaseline/fully_connected_2/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
"baseline/fully_connected_2/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *5
_class+
)'loc:@baseline/fully_connected_2/weights*
shared_name 
�
)baseline/fully_connected_2/weights/AssignAssign"baseline/fully_connected_2/weights=baseline/fully_connected_2/weights/Initializer/random_uniform*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
'baseline/fully_connected_2/weights/readIdentity"baseline/fully_connected_2/weights*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
3baseline/fully_connected_2/biases/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
!baseline/fully_connected_2/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_2/biases*
shared_name 
�
(baseline/fully_connected_2/biases/AssignAssign!baseline/fully_connected_2/biases3baseline/fully_connected_2/biases/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
&baseline/fully_connected_2/biases/readIdentity!baseline/fully_connected_2/biases*4
_class*
(&loc:@baseline/fully_connected_2/biases*
T0*
_output_shapes
:
�
!baseline/fully_connected_2/MatMulMatMulbaseline/fully_connected_1/Relu'baseline/fully_connected_2/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
"baseline/fully_connected_2/BiasAddBiasAdd!baseline/fully_connected_2/MatMul&baseline/fully_connected_2/biases/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
e

baseline_1Placeholder*
dtype0*
shape:���������*#
_output_shapes
:���������
m
SqueezeSqueeze"baseline/fully_connected_2/BiasAdd*
squeeze_dims
 *
T0*
_output_shapes
:
q
$mean_squared_error/SquaredDifferenceSquaredDifference
baseline_1Squeeze*
T0*
_output_shapes
:
t
/mean_squared_error/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
dtype0*
valueB *
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
�
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:���������
�
3mean_squared_error/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference*
T0*
_output_shapes
: 
u
3mean_squared_error/assert_broadcastable/is_scalar/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
1mean_squared_error/assert_broadcastable/is_scalarEqual3mean_squared_error/assert_broadcastable/is_scalar/x4mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
=mean_squared_error/assert_broadcastable/is_valid_shape/SwitchSwitch1mean_squared_error/assert_broadcastable/is_scalar1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_tIdentity?mean_squared_error/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_fIdentity=mean_squared_error/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
>mean_squared_error/assert_broadcastable/is_valid_shape/pred_idIdentity1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
?mean_squared_error/assert_broadcastable/is_valid_shape/Switch_1Switch1mean_squared_error/assert_broadcastable/is_scalar>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualdmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchfmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
dmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch3mean_squared_error/assert_broadcastable/values/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*F
_class<
:8loc:@mean_squared_error/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
�
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch4mean_squared_error/assert_broadcastable/weights/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
�
Wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityWmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimswmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch4mean_squared_error/assert_broadcastable/values/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchsmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapelmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
�
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:���������
�
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimskmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0
�
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
�
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch5mean_squared_error/assert_broadcastable/weights/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchumean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationnmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*<
_output_shapes*
(:���������:���������:*
T0*
set_operationa-b
�
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
�
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
�
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualcmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xrmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*p
_classf
dbloc:@mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
Vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N
�
<mean_squared_error/assert_broadcastable/is_valid_shape/MergeMergeVmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeAmean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
�
-mean_squared_error/assert_broadcastable/ConstConst*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
~
/mean_squared_error/assert_broadcastable/Const_1Const*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
/mean_squared_error/assert_broadcastable/Const_2Const*
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
_output_shapes
: 
}
/mean_squared_error/assert_broadcastable/Const_3Const*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
/mean_squared_error/assert_broadcastable/Const_4Const*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
z
/mean_squared_error/assert_broadcastable/Const_5Const*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
:mean_squared_error/assert_broadcastable/AssertGuard/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
<mean_squared_error/assert_broadcastable/AssertGuard/switch_tIdentity<mean_squared_error/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
<mean_squared_error/assert_broadcastable/AssertGuard/switch_fIdentity:mean_squared_error/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
;mean_squared_error/assert_broadcastable/AssertGuard/pred_idIdentity<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 

8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp=^mean_squared_error/assert_broadcastable/AssertGuard/switch_t
�
Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependencyIdentity<mean_squared_error/assert_broadcastable/AssertGuard/switch_t9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertAmean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchAmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Switch5mean_squared_error/assert_broadcastable/weights/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Switch4mean_squared_error/assert_broadcastable/values/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3Switch1mean_squared_error/assert_broadcastable/is_scalar;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Identity<mean_squared_error/assert_broadcastable/AssertGuard/switch_f;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
�
9mean_squared_error/assert_broadcastable/AssertGuard/MergeMergeHmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
�
mean_squared_error/ToFloat_3/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
X
mean_squared_error/RankRankmean_squared_error/Mul*
T0*
_output_shapes
: 
�
mean_squared_error/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
mean_squared_error/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
�
mean_squared_error/rangeRangemean_squared_error/range/startmean_squared_error/Rankmean_squared_error/range/delta*

Tidx0*#
_output_shapes
:���������
�
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/range*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
&mean_squared_error/num_present/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat_3/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
�
)mean_squared_error/num_present/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
.mean_squared_error/num_present/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
.mean_squared_error/num_present/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
�
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:���������
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
: 
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*d
_classZ
XVloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
�
umean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityumean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShape�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*<
_output_shapes*
(:���������:���������:*
T0*
set_operationa-b
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
�
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*�
_class�
��loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergewmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N
�
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergetmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
�
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
�
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
�
dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
�
Wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergefmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:���������
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*
_output_shapes
:
�
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
~
#mean_squared_error/num_present/RankRank0mean_squared_error/num_present/broadcast_weights*
T0*
_output_shapes
: 
�
*mean_squared_error/num_present/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
*mean_squared_error/num_present/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
�
$mean_squared_error/num_present/rangeRange*mean_squared_error/num_present/range/start#mean_squared_error/num_present/Rank*mean_squared_error/num_present/range/delta*

Tidx0*#
_output_shapes
:���������
�
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/range*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
mean_squared_error/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
mean_squared_error/Greater/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
T0*
_output_shapes
: 
�
mean_squared_error/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
~
mean_squared_error/EqualEqualmean_squared_error/num_presentmean_squared_error/Equal/y*
T0*
_output_shapes
: 
�
"mean_squared_error/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
"mean_squared_error/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*
_output_shapes
: 
�
mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
�
mean_squared_error/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
mean_squared_error/valueSelectmean_squared_error/Greatermean_squared_error/divmean_squared_error/zeros_like*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
c
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*
_output_shapes
: 
y
4gradients_1/mean_squared_error/value_grad/zeros_likeConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
0gradients_1/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients_1/Fill4gradients_1/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
�
2gradients_1/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater4gradients_1/mean_squared_error/value_grad/zeros_likegradients_1/Fill*
T0*
_output_shapes
: 
�
:gradients_1/mean_squared_error/value_grad/tuple/group_depsNoOp1^gradients_1/mean_squared_error/value_grad/Select3^gradients_1/mean_squared_error/value_grad/Select_1
�
Bgradients_1/mean_squared_error/value_grad/tuple/control_dependencyIdentity0gradients_1/mean_squared_error/value_grad/Select;^gradients_1/mean_squared_error/value_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/mean_squared_error/value_grad/Select*
T0*
_output_shapes
: 
�
Dgradients_1/mean_squared_error/value_grad/tuple/control_dependency_1Identity2gradients_1/mean_squared_error/value_grad/Select_1;^gradients_1/mean_squared_error/value_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/mean_squared_error/value_grad/Select_1*
T0*
_output_shapes
: 
p
-gradients_1/mean_squared_error/div_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
r
/gradients_1/mean_squared_error/div_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/div_grad/Shape/gradients_1/mean_squared_error/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/gradients_1/mean_squared_error/div_grad/RealDivRealDivBgradients_1/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
�
+gradients_1/mean_squared_error/div_grad/SumSum/gradients_1/mean_squared_error/div_grad/RealDiv=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients_1/mean_squared_error/div_grad/ReshapeReshape+gradients_1/mean_squared_error/div_grad/Sum-gradients_1/mean_squared_error/div_grad/Shape*
_output_shapes
: *
T0*
Tshape0
m
+gradients_1/mean_squared_error/div_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
�
1gradients_1/mean_squared_error/div_grad/RealDiv_1RealDiv+gradients_1/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
�
1gradients_1/mean_squared_error/div_grad/RealDiv_2RealDiv1gradients_1/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
�
+gradients_1/mean_squared_error/div_grad/mulMulBgradients_1/mean_squared_error/value_grad/tuple/control_dependency1gradients_1/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
-gradients_1/mean_squared_error/div_grad/Sum_1Sum+gradients_1/mean_squared_error/div_grad/mul?gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
1gradients_1/mean_squared_error/div_grad/Reshape_1Reshape-gradients_1/mean_squared_error/div_grad/Sum_1/gradients_1/mean_squared_error/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
8gradients_1/mean_squared_error/div_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/div_grad/Reshape2^gradients_1/mean_squared_error/div_grad/Reshape_1
�
@gradients_1/mean_squared_error/div_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/div_grad/Reshape9^gradients_1/mean_squared_error/div_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/div_grad/Reshape*
T0*
_output_shapes
: 
�
Bgradients_1/mean_squared_error/div_grad/tuple/control_dependency_1Identity1gradients_1/mean_squared_error/div_grad/Reshape_19^gradients_1/mean_squared_error/div_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/div_grad/Reshape_1*
T0*
_output_shapes
: 
z
7gradients_1/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
1gradients_1/mean_squared_error/Sum_1_grad/ReshapeReshape@gradients_1/mean_squared_error/div_grad/tuple/control_dependency7gradients_1/mean_squared_error/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
{
8gradients_1/mean_squared_error/Sum_1_grad/Tile/multiplesConst*
dtype0*
valueB *
_output_shapes
: 
�
.gradients_1/mean_squared_error/Sum_1_grad/TileTile1gradients_1/mean_squared_error/Sum_1_grad/Reshape8gradients_1/mean_squared_error/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
z
5gradients_1/mean_squared_error/Select_grad/zeros_likeConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
1gradients_1/mean_squared_error/Select_grad/SelectSelectmean_squared_error/EqualBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_15gradients_1/mean_squared_error/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
3gradients_1/mean_squared_error/Select_grad/Select_1Selectmean_squared_error/Equal5gradients_1/mean_squared_error/Select_grad/zeros_likeBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
;gradients_1/mean_squared_error/Select_grad/tuple/group_depsNoOp2^gradients_1/mean_squared_error/Select_grad/Select4^gradients_1/mean_squared_error/Select_grad/Select_1
�
Cgradients_1/mean_squared_error/Select_grad/tuple/control_dependencyIdentity1gradients_1/mean_squared_error/Select_grad/Select<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Select_grad/Select*
T0*
_output_shapes
: 
�
Egradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1Identity3gradients_1/mean_squared_error/Select_grad/Select_1<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*F
_class<
:8loc:@gradients_1/mean_squared_error/Select_grad/Select_1*
T0*
_output_shapes
: 
�
-gradients_1/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
out_type0*
T0*#
_output_shapes
:���������
�
,gradients_1/mean_squared_error/Sum_grad/SizeSize-gradients_1/mean_squared_error/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
: *@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape
�
+gradients_1/mean_squared_error/Sum_grad/addAddmean_squared_error/range,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
+gradients_1/mean_squared_error/Sum_grad/modFloorMod+gradients_1/mean_squared_error/Sum_grad/add,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
/gradients_1/mean_squared_error/Sum_grad/Shape_1Shape+gradients_1/mean_squared_error/Sum_grad/mod*
out_type0*
T0*
_output_shapes
:*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape
�
3gradients_1/mean_squared_error/Sum_grad/range/startConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B : *
_output_shapes
: 
�
3gradients_1/mean_squared_error/Sum_grad/range/deltaConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
-gradients_1/mean_squared_error/Sum_grad/rangeRange3gradients_1/mean_squared_error/Sum_grad/range/start,gradients_1/mean_squared_error/Sum_grad/Size3gradients_1/mean_squared_error/Sum_grad/range/delta*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*

Tidx0*#
_output_shapes
:���������
�
2gradients_1/mean_squared_error/Sum_grad/Fill/valueConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
,gradients_1/mean_squared_error/Sum_grad/FillFill/gradients_1/mean_squared_error/Sum_grad/Shape_12gradients_1/mean_squared_error/Sum_grad/Fill/value*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
5gradients_1/mean_squared_error/Sum_grad/DynamicStitchDynamicStitch-gradients_1/mean_squared_error/Sum_grad/range+gradients_1/mean_squared_error/Sum_grad/mod-gradients_1/mean_squared_error/Sum_grad/Shape,gradients_1/mean_squared_error/Sum_grad/Fill*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������*
N
�
1gradients_1/mean_squared_error/Sum_grad/Maximum/yConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
/gradients_1/mean_squared_error/Sum_grad/MaximumMaximum5gradients_1/mean_squared_error/Sum_grad/DynamicStitch1gradients_1/mean_squared_error/Sum_grad/Maximum/y*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
0gradients_1/mean_squared_error/Sum_grad/floordivFloorDiv-gradients_1/mean_squared_error/Sum_grad/Shape/gradients_1/mean_squared_error/Sum_grad/Maximum*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
/gradients_1/mean_squared_error/Sum_grad/ReshapeReshape.gradients_1/mean_squared_error/Sum_1_grad/Tile5gradients_1/mean_squared_error/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
,gradients_1/mean_squared_error/Sum_grad/TileTile/gradients_1/mean_squared_error/Sum_grad/Reshape0gradients_1/mean_squared_error/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
�
5gradients_1/mean_squared_error/num_present_grad/ShapeShape0mean_squared_error/num_present/broadcast_weights*
out_type0*
T0*#
_output_shapes
:���������
�
4gradients_1/mean_squared_error/num_present_grad/SizeSize5gradients_1/mean_squared_error/num_present_grad/Shape*
out_type0*
T0*
_output_shapes
: *H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape
�
3gradients_1/mean_squared_error/num_present_grad/addAdd$mean_squared_error/num_present/range4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
3gradients_1/mean_squared_error/num_present_grad/modFloorMod3gradients_1/mean_squared_error/num_present_grad/add4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
7gradients_1/mean_squared_error/num_present_grad/Shape_1Shape3gradients_1/mean_squared_error/num_present_grad/mod*
out_type0*
T0*
_output_shapes
:*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape
�
;gradients_1/mean_squared_error/num_present_grad/range/startConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B : *
_output_shapes
: 
�
;gradients_1/mean_squared_error/num_present_grad/range/deltaConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
�
5gradients_1/mean_squared_error/num_present_grad/rangeRange;gradients_1/mean_squared_error/num_present_grad/range/start4gradients_1/mean_squared_error/num_present_grad/Size;gradients_1/mean_squared_error/num_present_grad/range/delta*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*

Tidx0*#
_output_shapes
:���������
�
:gradients_1/mean_squared_error/num_present_grad/Fill/valueConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
�
4gradients_1/mean_squared_error/num_present_grad/FillFill7gradients_1/mean_squared_error/num_present_grad/Shape_1:gradients_1/mean_squared_error/num_present_grad/Fill/value*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
=gradients_1/mean_squared_error/num_present_grad/DynamicStitchDynamicStitch5gradients_1/mean_squared_error/num_present_grad/range3gradients_1/mean_squared_error/num_present_grad/mod5gradients_1/mean_squared_error/num_present_grad/Shape4gradients_1/mean_squared_error/num_present_grad/Fill*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������*
N
�
9gradients_1/mean_squared_error/num_present_grad/Maximum/yConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
�
7gradients_1/mean_squared_error/num_present_grad/MaximumMaximum=gradients_1/mean_squared_error/num_present_grad/DynamicStitch9gradients_1/mean_squared_error/num_present_grad/Maximum/y*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
8gradients_1/mean_squared_error/num_present_grad/floordivFloorDiv5gradients_1/mean_squared_error/num_present_grad/Shape7gradients_1/mean_squared_error/num_present_grad/Maximum*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
7gradients_1/mean_squared_error/num_present_grad/ReshapeReshapeEgradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1=gradients_1/mean_squared_error/num_present_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
4gradients_1/mean_squared_error/num_present_grad/TileTile7gradients_1/mean_squared_error/num_present_grad/Reshape8gradients_1/mean_squared_error/num_present_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
�
-gradients_1/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:���������
r
/gradients_1/mean_squared_error/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/Mul_grad/Shape/gradients_1/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients_1/mean_squared_error/Mul_grad/mulMul,gradients_1/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
�
+gradients_1/mean_squared_error/Mul_grad/SumSum+gradients_1/mean_squared_error/Mul_grad/mul=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients_1/mean_squared_error/Mul_grad/ReshapeReshape+gradients_1/mean_squared_error/Mul_grad/Sum-gradients_1/mean_squared_error/Mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
-gradients_1/mean_squared_error/Mul_grad/mul_1Mul$mean_squared_error/SquaredDifference,gradients_1/mean_squared_error/Sum_grad/Tile*
T0*
_output_shapes
:
�
-gradients_1/mean_squared_error/Mul_grad/Sum_1Sum-gradients_1/mean_squared_error/Mul_grad/mul_1?gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
1gradients_1/mean_squared_error/Mul_grad/Reshape_1Reshape-gradients_1/mean_squared_error/Mul_grad/Sum_1/gradients_1/mean_squared_error/Mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
8gradients_1/mean_squared_error/Mul_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/Mul_grad/Reshape2^gradients_1/mean_squared_error/Mul_grad/Reshape_1
�
@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/Mul_grad/Reshape9^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/Mul_grad/Reshape*
T0*
_output_shapes
:
�
Bgradients_1/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity1gradients_1/mean_squared_error/Mul_grad/Reshape_19^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Mul_grad/Reshape_1*
T0*
_output_shapes
: 
�
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1Shape:mean_squared_error/num_present/broadcast_weights/ones_like*
out_type0*
T0*#
_output_shapes
:���������
�
Wgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulMul4gradients_1/mean_squared_error/num_present_grad/Tile:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
�
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumSumEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulWgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeReshapeEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Mul%mean_squared_error/num_present/Select4gradients_1/mean_squared_error/num_present_grad/Tile*
T0*
_output_shapes
:
�
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Ygradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Kgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1ReshapeGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
Rgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_depsNoOpJ^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeL^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1
�
Zgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeS^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape*
T0*
_output_shapes
: 
�
\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityKgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1S^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1*
T0*
_output_shapes
:
�
Pgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankRank\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/rangeRangeWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startPgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/delta*

Tidx0*#
_output_shapes
:���������
�
Ogradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/SumSum\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
;gradients_1/mean_squared_error/SquaredDifference_grad/ShapeShape
baseline_1*
out_type0*
T0*
_output_shapes
:
�
=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1ShapeSqueeze*
out_type0*
T0*#
_output_shapes
:���������
�
Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/mean_squared_error/SquaredDifference_grad/Shape=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients_1/mean_squared_error/SquaredDifference_grad/scalarConstA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
9gradients_1/mean_squared_error/SquaredDifference_grad/mulMul<gradients_1/mean_squared_error/SquaredDifference_grad/scalar@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
9gradients_1/mean_squared_error/SquaredDifference_grad/subSub
baseline_1SqueezeA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mul9gradients_1/mean_squared_error/SquaredDifference_grad/mul9gradients_1/mean_squared_error/SquaredDifference_grad/sub*
T0*
_output_shapes
:
�
9gradients_1/mean_squared_error/SquaredDifference_grad/SumSum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeReshape9gradients_1/mean_squared_error/SquaredDifference_grad/Sum;gradients_1/mean_squared_error/SquaredDifference_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1Sum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
9gradients_1/mean_squared_error/SquaredDifference_grad/NegNeg?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes
:
�
Fgradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp>^gradients_1/mean_squared_error/SquaredDifference_grad/Reshape:^gradients_1/mean_squared_error/SquaredDifference_grad/Neg
�
Ngradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/mean_squared_error/SquaredDifference_grad/Reshape*
T0*#
_output_shapes
:���������
�
Pgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity9gradients_1/mean_squared_error/SquaredDifference_grad/NegG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/mean_squared_error/SquaredDifference_grad/Neg*
T0*
_output_shapes
:
�
gradients_1/Squeeze_grad/ShapeShape"baseline/fully_connected_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
 gradients_1/Squeeze_grad/ReshapeReshapePgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1gradients_1/Squeeze_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
?gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients_1/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
�
Dgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients_1/Squeeze_grad/Reshape@^gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGrad
�
Lgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients_1/Squeeze_grad/ReshapeE^gradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Squeeze_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ngradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGradE^gradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
9gradients_1/baseline/fully_connected_2/MatMul_grad/MatMulMatMulLgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency'baseline/fully_connected_2/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
;gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1MatMulbaseline/fully_connected_1/ReluLgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Cgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul<^gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1
�
Kgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_2/MatMul_grad/MatMulD^gradients_1/baseline/fully_connected_2/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Mgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependency_1Identity;gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1D^gradients_1/baseline/fully_connected_2/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
9gradients_1/baseline/fully_connected_1/Relu_grad/ReluGradReluGradKgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependencybaseline/fully_connected_1/Relu*
T0*'
_output_shapes
:��������� 
�
?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/baseline/fully_connected_1/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
�
Dgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_1/Relu_grad/ReluGrad@^gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad
�
Lgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_1/Relu_grad/ReluGradE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Ngradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulMatMulLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency'baseline/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1MatMulbaseline/fully_connected/ReluLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:  
�
Cgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul<^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1
�
Kgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulD^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Mgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1D^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
�
7gradients_1/baseline/fully_connected/Relu_grad/ReluGradReluGradKgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencybaseline/fully_connected/Relu*
T0*'
_output_shapes
:��������� 
�
=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
�
Bgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/Relu_grad/ReluGrad>^gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad
�
Jgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/Relu_grad/ReluGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Lgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
7gradients_1/baseline/fully_connected/MatMul_grad/MatMulMatMulJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency%baseline/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1MatMuloJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Agradients_1/baseline/fully_connected/MatMul_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/MatMul_grad/MatMul:^gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1
�
Igradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/MatMul_grad/MatMulB^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Kgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1Identity9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1B^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
beta1_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta1_power_1/readIdentitybeta1_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
beta2_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta2_power_1/readIdentitybeta2_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
7baseline/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
%baseline/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
�
,baseline/fully_connected/weights/Adam/AssignAssign%baseline/fully_connected/weights/Adam7baseline/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

: 
�
*baseline/fully_connected/weights/Adam/readIdentity%baseline/fully_connected/weights/Adam*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
9baseline/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
'baseline/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
�
.baseline/fully_connected/weights/Adam_1/AssignAssign'baseline/fully_connected/weights/Adam_19baseline/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

: 
�
,baseline/fully_connected/weights/Adam_1/readIdentity'baseline/fully_connected/weights/Adam_1*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
6baseline/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
$baseline/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
+baseline/fully_connected/biases/Adam/AssignAssign$baseline/fully_connected/biases/Adam6baseline/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
)baseline/fully_connected/biases/Adam/readIdentity$baseline/fully_connected/biases/Adam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
8baseline/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
&baseline/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
-baseline/fully_connected/biases/Adam_1/AssignAssign&baseline/fully_connected/biases/Adam_18baseline/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
+baseline/fully_connected/biases/Adam_1/readIdentity&baseline/fully_connected/biases/Adam_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
9baseline/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
'baseline/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
�
.baseline/fully_connected_1/weights/Adam/AssignAssign'baseline/fully_connected_1/weights/Adam9baseline/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:  
�
,baseline/fully_connected_1/weights/Adam/readIdentity'baseline/fully_connected_1/weights/Adam*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
;baseline/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
)baseline/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
�
0baseline/fully_connected_1/weights/Adam_1/AssignAssign)baseline/fully_connected_1/weights/Adam_1;baseline/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:  
�
.baseline/fully_connected_1/weights/Adam_1/readIdentity)baseline/fully_connected_1/weights/Adam_1*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
8baseline/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
&baseline/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
�
-baseline/fully_connected_1/biases/Adam/AssignAssign&baseline/fully_connected_1/biases/Adam8baseline/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
: 
�
+baseline/fully_connected_1/biases/Adam/readIdentity&baseline/fully_connected_1/biases/Adam*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
: 
�
:baseline/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
(baseline/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
�
/baseline/fully_connected_1/biases/Adam_1/AssignAssign(baseline/fully_connected_1/biases/Adam_1:baseline/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
: 
�
-baseline/fully_connected_1/biases/Adam_1/readIdentity(baseline/fully_connected_1/biases/Adam_1*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
: 
�
9baseline/fully_connected_2/weights/Adam/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
'baseline/fully_connected_2/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *5
_class+
)'loc:@baseline/fully_connected_2/weights*
shared_name 
�
.baseline/fully_connected_2/weights/Adam/AssignAssign'baseline/fully_connected_2/weights/Adam9baseline/fully_connected_2/weights/Adam/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
,baseline/fully_connected_2/weights/Adam/readIdentity'baseline/fully_connected_2/weights/Adam*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
;baseline/fully_connected_2/weights/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
)baseline/fully_connected_2/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *5
_class+
)'loc:@baseline/fully_connected_2/weights*
shared_name 
�
0baseline/fully_connected_2/weights/Adam_1/AssignAssign)baseline/fully_connected_2/weights/Adam_1;baseline/fully_connected_2/weights/Adam_1/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
.baseline/fully_connected_2/weights/Adam_1/readIdentity)baseline/fully_connected_2/weights/Adam_1*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
8baseline/fully_connected_2/biases/Adam/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
&baseline/fully_connected_2/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_2/biases*
shared_name 
�
-baseline/fully_connected_2/biases/Adam/AssignAssign&baseline/fully_connected_2/biases/Adam8baseline/fully_connected_2/biases/Adam/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
+baseline/fully_connected_2/biases/Adam/readIdentity&baseline/fully_connected_2/biases/Adam*4
_class*
(&loc:@baseline/fully_connected_2/biases*
T0*
_output_shapes
:
�
:baseline/fully_connected_2/biases/Adam_1/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
(baseline/fully_connected_2/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_2/biases*
shared_name 
�
/baseline/fully_connected_2/biases/Adam_1/AssignAssign(baseline/fully_connected_2/biases/Adam_1:baseline/fully_connected_2/biases/Adam_1/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
-baseline/fully_connected_2/biases/Adam_1/readIdentity(baseline/fully_connected_2/biases/Adam_1*4
_class*
(&loc:@baseline/fully_connected_2/biases*
T0*
_output_shapes
:
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *���<*
_output_shapes
: 
Q
Adam_1/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Q
Adam_1/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
8Adam_1/update_baseline/fully_connected/weights/ApplyAdam	ApplyAdam baseline/fully_connected/weights%baseline/fully_connected/weights/Adam'baseline/fully_connected/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonKgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking( *
T0*
_output_shapes

: 
�
7Adam_1/update_baseline/fully_connected/biases/ApplyAdam	ApplyAdambaseline/fully_connected/biases$baseline/fully_connected/biases/Adam&baseline/fully_connected/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonLgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
:Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam	ApplyAdam"baseline/fully_connected_1/weights'baseline/fully_connected_1/weights/Adam)baseline/fully_connected_1/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonMgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:  
�
9Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam	ApplyAdam!baseline/fully_connected_1/biases&baseline/fully_connected_1/biases/Adam(baseline/fully_connected_1/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
: 
�
:Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam	ApplyAdam"baseline/fully_connected_2/weights'baseline/fully_connected_2/weights/Adam)baseline/fully_connected_2/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonMgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking( *
T0*
_output_shapes

: 
�
9Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam	ApplyAdam!baseline/fully_connected_2/biases&baseline/fully_connected_2/biases/Adam(baseline/fully_connected_2/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking( *
T0*
_output_shapes
:
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta19^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta29^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
Adam_1NoOp9^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam^Adam_1/Assign^Adam_1/Assign_1
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
N"iD�"�+     P��u	���Ң�AJ��
�&�&
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
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
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
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
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
,
Exp
x"T
y"T"
Ttype:

2
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
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
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
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
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
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sign
x"T
y"T"
Ttype:

2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.5.02v1.5.0-0-g37aa430d84��
d
oPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
{
a_contPlaceholder*
dtype0*%
shape:������������������*0
_output_shapes
:������������������
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
valueB"       *
_output_shapes
:
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�)��*
_output_shapes
: 
�
Epolicy_network/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *�)�>*
_output_shapes
: 
�
Opolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformGpolicy_network/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

: *
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

: 
�
Apolicy_network/fully_connected/weights/Initializer/random_uniformAddEpolicy_network/fully_connected/weights/Initializer/random_uniform/mulEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
&policy_network/fully_connected/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *9
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

: 
�
+policy_network/fully_connected/weights/readIdentity&policy_network/fully_connected/weights*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
7policy_network/fully_connected/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
%policy_network/fully_connected/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
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
: 
�
*policy_network/fully_connected/biases/readIdentity%policy_network/fully_connected/biases*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
%policy_network/fully_connected/MatMulMatMulo+policy_network/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
&policy_network/fully_connected/BiasAddBiasAdd%policy_network/fully_connected/MatMul*policy_network/fully_connected/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:��������� 
�
#policy_network/fully_connected/ReluRelu&policy_network/fully_connected/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Ipolicy_network/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB"        *
_output_shapes
:
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *qĜ�*
_output_shapes
: 
�
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *qĜ>*
_output_shapes
: 
�
Qpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:  *
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

:  
�
Cpolicy_network/fully_connected_1/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
(policy_network/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *;
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

:  
�
-policy_network/fully_connected_1/weights/readIdentity(policy_network/fully_connected_1/weights*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
9policy_network/fully_connected_1/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
'policy_network/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *:
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
: 
�
,policy_network/fully_connected_1/biases/readIdentity'policy_network/fully_connected_1/biases*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
: 
�
'policy_network/fully_connected_1/MatMulMatMul#policy_network/fully_connected/Relu-policy_network/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
(policy_network/fully_connected_1/BiasAddBiasAdd'policy_network/fully_connected_1/MatMul,policy_network/fully_connected_1/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:��������� 
�
%policy_network/fully_connected_1/ReluRelu(policy_network/fully_connected_1/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Ipolicy_network/fully_connected_2/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB"       *
_output_shapes
:
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB
 *�r˾*
_output_shapes
: 
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB
 *�r�>*
_output_shapes
: 
�
Qpolicy_network/fully_connected_2/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_2/weights/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*;
_class1
/-loc:@policy_network/fully_connected_2/weights
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/subSubGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/maxGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes
: 
�
Gpolicy_network/fully_connected_2/weights/Initializer/random_uniform/mulMulQpolicy_network/fully_connected_2/weights/Initializer/random_uniform/RandomUniformGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
Cpolicy_network/fully_connected_2/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_2/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
(policy_network/fully_connected_2/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
shared_name 
�
/policy_network/fully_connected_2/weights/AssignAssign(policy_network/fully_connected_2/weightsCpolicy_network/fully_connected_2/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
-policy_network/fully_connected_2/weights/readIdentity(policy_network/fully_connected_2/weights*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
9policy_network/fully_connected_2/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
'policy_network/fully_connected_2/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
shared_name 
�
.policy_network/fully_connected_2/biases/AssignAssign'policy_network/fully_connected_2/biases9policy_network/fully_connected_2/biases/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
,policy_network/fully_connected_2/biases/readIdentity'policy_network/fully_connected_2/biases*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
T0*
_output_shapes
:
�
'policy_network/fully_connected_2/MatMulMatMul%policy_network/fully_connected_1/Relu-policy_network/fully_connected_2/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
(policy_network/fully_connected_2/BiasAddBiasAdd'policy_network/fully_connected_2/MatMul,policy_network/fully_connected_2/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
�
(log_std/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@log_std*
valueB:*
_output_shapes
:
�
&log_std/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@log_std*
valueB
 *�5�*
_output_shapes
: 
�
&log_std/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@log_std*
valueB
 *�5?*
_output_shapes
: 
�
0log_std/Initializer/random_uniform/RandomUniformRandomUniform(log_std/Initializer/random_uniform/shape*
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*
_class
loc:@log_std
�
&log_std/Initializer/random_uniform/subSub&log_std/Initializer/random_uniform/max&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
&log_std/Initializer/random_uniform/mulMul0log_std/Initializer/random_uniform/RandomUniform&log_std/Initializer/random_uniform/sub*
_class
loc:@log_std*
T0*
_output_shapes
:
�
"log_std/Initializer/random_uniformAdd&log_std/Initializer/random_uniform/mul&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
:
�
log_std
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
�
log_std/AssignAssignlog_std"log_std/Initializer/random_uniform*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
b
log_std/readIdentitylog_std*
_class
loc:@log_std*
T0*
_output_shapes
:
=
ExpExplog_std/read*
T0*
_output_shapes
:
d
random_normal/shapeConst*
dtype0*
valueB"�a     *
_output_shapes
:
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
l
random_normal/mulMul"random_normal/RandomStandardNormalExp*
T0* 
_output_shapes
:
��
|
random_normalAddrandom_normal/mul(policy_network/fully_connected_2/BiasAdd*
T0* 
_output_shapes
:
��
?
Exp_1Explog_std/read*
T0*
_output_shapes
:
�
bMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
�
fMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag_1/batch_shape_tensor/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
8MultivariateNormalDiag_1/shapes_from_loc_and_scale/ShapeShape(policy_network/fully_connected_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
FMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_1Const*
dtype0*
valueB:
���������*
_output_shapes
:
�
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
@MultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_sliceStridedSlice8MultivariateNormalDiag_1/shapes_from_loc_and_scale/ShapeFMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stackHMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_1HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
^MultivariateNormalDiag_1/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsfMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag_1/batch_shape_tensor/batch_shape@MultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice*
T0*
_output_shapes
:
J
zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
I
onesConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
>

Normal/locIdentityzeros*
T0*
_output_shapes
: 
?
Normal/scaleIdentityones*
T0*
_output_shapes
: 
i
'affine_linear_operator/init/event_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
>affine_linear_operator/init/DistributionShape/init/batch_ndimsConst*
dtype0*
value	B : *
_output_shapes
: 
�
>affine_linear_operator/init/DistributionShape/init/event_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
c
!affine_linear_operator/init/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
_
MultivariateNormalDiag_2/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
a
MultivariateNormalDiag_2/emptyConst*
dtype0*
valueB *
_output_shapes
: 
�
?MultivariateNormalDiag_2/Normal/is_scalar_batch/is_scalar_batchConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
`
MultivariateNormalDiag_2/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_1Const*
dtype0*
value	B : *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_2Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_3Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_4Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_5Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_6Const*
dtype0*
value	B :*
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_7Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
b
 MultivariateNormalDiag_2/Const_8Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
AMultivariateNormalDiag_2/Normal_1/is_scalar_event/is_scalar_eventConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
n
$MultivariateNormalDiag_2/event_shapeConst*
dtype0*
valueB:*
_output_shapes
:
b
 MultivariateNormalDiag_2/Const_9Const*
dtype0*
value	B :*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_10Const*
dtype0*
value	B : *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_11Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_12Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_13Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_14Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_15Const*
dtype0*
value	B :*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_16Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_17Const*
dtype0
*
value	B
 Z*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_18Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
AMultivariateNormalDiag_2/Normal_2/is_scalar_batch/is_scalar_batchConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_19Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_20Const*
dtype0
*
value	B
 Z *
_output_shapes
: 
c
!MultivariateNormalDiag_2/Const_21Const*
dtype0*
value	B :*
_output_shapes
: 
`
MultivariateNormalDiag_2/sub/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
MultivariateNormalDiag_2/subSubMultivariateNormalDiag_2/sub/x!MultivariateNormalDiag_2/Const_21*
T0*
_output_shapes
: 
f
$MultivariateNormalDiag_2/range/limitConst*
dtype0*
value	B : *
_output_shapes
: 
f
$MultivariateNormalDiag_2/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
MultivariateNormalDiag_2/rangeRangeMultivariateNormalDiag_2/sub$MultivariateNormalDiag_2/range/limit$MultivariateNormalDiag_2/range/delta*

Tidx0*#
_output_shapes
:���������
�
DMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/subSuba_cont(policy_network/fully_connected_2/BiasAdd*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAdd�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/x�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
_output_shapes
:*

axis *
T0*
N
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
_output_shapes
:*

axis *
T0*
N
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSlice�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/begin�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/pick_vector/condConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concatConcatV2�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/ReshapeReshapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat*
Tshape0*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*'
_output_shapes
:���������
�
cMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
aMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivRealDivcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xExp_1*
T0*
_output_shapes
:
�
hMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
dMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims
ExpandDimsaMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivhMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mulMul�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*'
_output_shapes
:���������
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAdd�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/x�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
_output_shapes
:*

axis *
T0*
N
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePack�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
_output_shapes
:*

axis *
T0*
N
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSlice�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/begin�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concatConcatV2�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/event_shape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/ReshapeReshape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat*
Tshape0*
T0*'
_output_shapes
:���������
�
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsAbsExp_1*
T0*
_output_shapes
:
�
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogLogvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs*
T0*
_output_shapes
:
�
�MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/SumSumvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log�MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
WMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/NegNegvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum*
T0*
_output_shapes
: 
�
AMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subSub�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape
Normal/loc*
T0*'
_output_shapes
:���������
�
EMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truedivRealDivAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subNormal/scale*
T0*'
_output_shapes
:���������
�
8MultivariateNormalDiag_3/log_prob/Normal/log_prob/SquareSquareEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/xConst*
dtype0*
valueB
 *   �*
_output_shapes
: 
�
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mulMul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*'
_output_shapes
:���������
k
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/LogLogNormal/scale*
T0*
_output_shapes
: 
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/xConst*
dtype0*
valueB
 *�?k?*
_output_shapes
: 
�
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/addAdd7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/x5MultivariateNormalDiag_3/log_prob/Normal/log_prob/Log*
T0*
_output_shapes
: 
�
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subSub5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul5MultivariateNormalDiag_3/log_prob/Normal/log_prob/add*
T0*'
_output_shapes
:���������
�
%MultivariateNormalDiag_3/log_prob/SumSum5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subMultivariateNormalDiag_2/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
%MultivariateNormalDiag_3/log_prob/addAdd%MultivariateNormalDiag_3/log_prob/SumWMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg*
T0*#
_output_shapes
:���������
_
NegNeg%MultivariateNormalDiag_3/log_prob/add*
T0*#
_output_shapes
:���������
D
mulMulNegadvtg*
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
[
gradients/mul_grad/ShapeShapeNeg*
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
b
gradients/mul_grad/mul_1MulNeggradients/Fill*
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
x
gradients/Neg_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*#
_output_shapes
:���������
�
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/ShapeShape%MultivariateNormalDiag_3/log_prob/Sum*
out_type0*
T0*#
_output_shapes
:���������

<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Jgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/MultivariateNormalDiag_3/log_prob/add_grad/SumSumgradients/Neg_grad/NegJgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeReshape8gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:
�
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1Sumgradients/Neg_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1Reshape:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
Egradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_depsNoOp=^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape?^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1
�
Mgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyIdentity<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeF^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*O
_classE
CAloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape*
T0*
_output_shapes
:
�
Ogradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1Identity>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1F^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub*
out_type0*
T0*
_output_shapes
:
�
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/SizeConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/addAddMultivariateNormalDiag_2/range9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/modFloorMod8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/add9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1Shape8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
:
�
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/startConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B : *
_output_shapes
: 
�
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/deltaConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/rangeRange@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/start9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/delta*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/valueConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/FillFill<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/value*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
Bgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitchDynamicStitch:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill*
N*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/yConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/MaximumMaximumBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/y*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordivFloorDiv:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*
_output_shapes
:
�
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ReshapeReshapeMgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
�
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileTile<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Reshape=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
jgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/NegNegOgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul*
out_type0*
T0*
_output_shapes
:
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumSum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1Sum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Tile\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegNegJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1ReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1
�
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1*
T0*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/SizeConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/addAdd�MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/modFloorMod�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/add�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1Const*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
valueB:*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/startConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B : *
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/deltaConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/rangeRange�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/start�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/delta*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/valueConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/FillFill�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/value*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitchDynamicStitch�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/mod�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill*
N*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/yConst*
dtype0*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/MaximumMaximum�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitch�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/y*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordivFloorDiv�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ReshapeReshapejgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/Neg�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/TileTile�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Reshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1Shape8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
out_type0*
T0*
_output_shapes
:
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulMul]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*'
_output_shapes
:���������
�
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumSumHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1Mul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1ReshapeJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1
�
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape*
T0*
_output_shapes
: 
�
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal
ReciprocalvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs�^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Tile*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mulMul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Tile�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal*
T0*
_output_shapes
:
�
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xConst`^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*'
_output_shapes
:���������
�
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Mul_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/SignSignExp_1*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mulMul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/Sign*
T0*
_output_shapes
:
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ShapeShapeAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
out_type0*
T0*
_output_shapes
:
�
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivRealDivMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Normal/scale*
T0*'
_output_shapes
:���������
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumSum\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivjgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ReshapeReshapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNegAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
T0*'
_output_shapes
:���������
�
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1RealDivXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNormal/scale*
T0*'
_output_shapes
:���������
�
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2RealDiv^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1Normal/scale*
T0*'
_output_shapes
:���������
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1SumXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mullgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1ReshapeZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
egradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_depsNoOp]^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1
�
mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentity\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshapef^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*o
_classe
caloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape*
T0*'
_output_shapes
:���������
�
ogradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependency_1Identity^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1f^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*q
_classg
ecloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
�
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape*
out_type0*
T0*
_output_shapes
:
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
fgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumSummgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyfgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ReshapeReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1Summgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyhgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegNegVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1ReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
agradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_depsNoOpY^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape[^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1
�
igradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependencyIdentityXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshapeb^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*k
_classa
_]loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1b^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1*
T0*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshapeigradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependency�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutation�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	Transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Reshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*'
_output_shapes
:���������
�
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ShapeShape�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgsBroadcastGradientArgsrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mulMul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*'
_output_shapes
:���������
�
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/SumSumpgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ReshapeReshapepgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1Mul�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose*
T0*'
_output_shapes
:���������
�
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1Reshapergradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:
�
}gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_depsNoOpu^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshapew^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependencyIdentitytgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*�
_class}
{yloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1Identityvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*�
_class
}{loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1*
T0*
_output_shapes

:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutation�MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	Transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*'
_output_shapes
:���������
�
ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeReshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Shape*
Tshape0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshape�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shapexgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDivRealDiv{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeExp_1*
T0*
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/SumSumxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ReshapeReshapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sumvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegNegcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/x*
T0*
_output_shapes
: 
�
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1RealDivtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegExp_1*
T0*
_output_shapes
:
�
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2RealDivzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1Exp_1*
T0*
_output_shapes
:
�
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mulMul{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Reshapezgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2*
T0*
_output_shapes
:
�
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1Sumtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1Reshapevgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_depsNoOpy^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape{^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependencyIdentityxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape�^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*�
_class�
}loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape*
T0*
_output_shapes
: 
�
�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1Identityzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1�^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*�
_class�
�loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1*
T0*
_output_shapes
:
�
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ShapeShapea_cont*
out_type0*
T0*
_output_shapes
:
�
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1Shape(policy_network/fully_connected_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
igradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgsBroadcastGradientArgsYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumSum�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapeigradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ReshapeReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape*
Tshape0*
T0*0
_output_shapes
:������������������
�
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1Sum�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapekgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/NegNegYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1*
T0*
_output_shapes
:
�
]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1ReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Neg[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
dgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_depsNoOp\^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape^^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1
�
lgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependencyIdentity[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshapee^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*n
_classd
b`loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape*
T0*0
_output_shapes
:������������������
�
ngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1Identity]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1e^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
gradients/AddNAddN�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mul�gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1*
N*�
_class�
��loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mul*
T0*
_output_shapes
:
[
gradients/Exp_1_grad/mulMulgradients/AddNExp_1*
T0*
_output_shapes
:
�
Cgradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGradBiasAddGradngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
T0*
_output_shapes
:
�
Hgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/group_depsNoOpo^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1D^gradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGrad
�
Pgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependencyIdentityngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1I^gradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Rgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
=gradients/policy_network/fully_connected_2/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_2/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
?gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1MatMul%policy_network/fully_connected_1/ReluPgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Ggradients/policy_network/fully_connected_2/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_2/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1
�
Ogradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_2/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_2/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Qgradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_2/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
=gradients/policy_network/fully_connected_1/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependency%policy_network/fully_connected_1/Relu*
T0*'
_output_shapes
:��������� 
�
Cgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad=gradients/policy_network/fully_connected_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
Hgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/Relu_grad/ReluGradD^gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad
�
Pgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/Relu_grad/ReluGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Rgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1MatMul#policy_network/fully_connected/ReluPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:  
�
Ggradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1
�
Ogradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Qgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
�
;gradients/policy_network/fully_connected/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency#policy_network/fully_connected/Relu*
T0*'
_output_shapes
:��������� 
�
Agradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
Fgradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/Relu_grad/ReluGradB^gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ngradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/Relu_grad/ReluGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Pgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
;gradients/policy_network/fully_connected/MatMul_grad/MatMulMatMulNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency+policy_network/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1MatMuloNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Egradients/policy_network/fully_connected/MatMul_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/MatMul_grad/MatMul>^gradients/policy_network/fully_connected/MatMul_grad/MatMul_1
�
Mgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/MatMul_grad/MatMulF^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Ogradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1Identity=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1F^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
z
beta1_power/initial_valueConst*
dtype0*
_class
loc:@log_std*
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
shape: *
_class
loc:@log_std*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
: 
f
beta1_power/readIdentitybeta1_power*
_class
loc:@log_std*
T0*
_output_shapes
: 
z
beta2_power/initial_valueConst*
dtype0*
_class
loc:@log_std*
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
shape: *
_class
loc:@log_std*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
: 
f
beta2_power/readIdentitybeta2_power*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
=policy_network/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
+policy_network/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *9
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

: 
�
0policy_network/fully_connected/weights/Adam/readIdentity+policy_network/fully_connected/weights/Adam*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
?policy_network/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
-policy_network/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *9
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

: 
�
2policy_network/fully_connected/weights/Adam_1/readIdentity-policy_network/fully_connected/weights/Adam_1*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

: 
�
<policy_network/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
*policy_network/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
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
: 
�
/policy_network/fully_connected/biases/Adam/readIdentity*policy_network/fully_connected/biases/Adam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
>policy_network/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
,policy_network/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *8
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
: 
�
1policy_network/fully_connected/biases/Adam_1/readIdentity,policy_network/fully_connected/biases/Adam_1*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
: 
�
?policy_network/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
-policy_network/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *;
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

:  
�
2policy_network/fully_connected_1/weights/Adam/readIdentity-policy_network/fully_connected_1/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
/policy_network/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *;
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

:  
�
4policy_network/fully_connected_1/weights/Adam_1/readIdentity/policy_network/fully_connected_1/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:  
�
>policy_network/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
,policy_network/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *:
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
: 
�
1policy_network/fully_connected_1/biases/Adam/readIdentity,policy_network/fully_connected_1/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
: 
�
@policy_network/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
.policy_network/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *:
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
: 
�
3policy_network/fully_connected_1/biases/Adam_1/readIdentity.policy_network/fully_connected_1/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
: 
�
?policy_network/fully_connected_2/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
-policy_network/fully_connected_2/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
shared_name 
�
4policy_network/fully_connected_2/weights/Adam/AssignAssign-policy_network/fully_connected_2/weights/Adam?policy_network/fully_connected_2/weights/Adam/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
2policy_network/fully_connected_2/weights/Adam/readIdentity-policy_network/fully_connected_2/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
Apolicy_network/fully_connected_2/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
/policy_network/fully_connected_2/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
shared_name 
�
6policy_network/fully_connected_2/weights/Adam_1/AssignAssign/policy_network/fully_connected_2/weights/Adam_1Apolicy_network/fully_connected_2/weights/Adam_1/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
4policy_network/fully_connected_2/weights/Adam_1/readIdentity/policy_network/fully_connected_2/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_2/weights*
T0*
_output_shapes

: 
�
>policy_network/fully_connected_2/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
,policy_network/fully_connected_2/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
shared_name 
�
3policy_network/fully_connected_2/biases/Adam/AssignAssign,policy_network/fully_connected_2/biases/Adam>policy_network/fully_connected_2/biases/Adam/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
1policy_network/fully_connected_2/biases/Adam/readIdentity,policy_network/fully_connected_2/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
T0*
_output_shapes
:
�
@policy_network/fully_connected_2/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
.policy_network/fully_connected_2/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
shared_name 
�
5policy_network/fully_connected_2/biases/Adam_1/AssignAssign.policy_network/fully_connected_2/biases/Adam_1@policy_network/fully_connected_2/biases/Adam_1/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
3policy_network/fully_connected_2/biases/Adam_1/readIdentity.policy_network/fully_connected_2/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_2/biases*
T0*
_output_shapes
:
�
log_std/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
�
log_std/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
�
log_std/Adam/AssignAssignlog_std/Adamlog_std/Adam/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
l
log_std/Adam/readIdentitylog_std/Adam*
_class
loc:@log_std*
T0*
_output_shapes
:
�
 log_std/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
�
log_std/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
�
log_std/Adam_1/AssignAssignlog_std/Adam_1 log_std/Adam_1/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
p
log_std/Adam_1/readIdentitylog_std/Adam_1*
_class
loc:@log_std*
T0*
_output_shapes
:
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

: 
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
: 
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

:  
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
: 
�
>Adam/update_policy_network/fully_connected_2/weights/ApplyAdam	ApplyAdam(policy_network/fully_connected_2/weights-policy_network/fully_connected_2/weights/Adam/policy_network/fully_connected_2/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/policy_network/fully_connected_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *;
_class1
/-loc:@policy_network/fully_connected_2/weights*
use_locking( *
T0*
_output_shapes

: 
�
=Adam/update_policy_network/fully_connected_2/biases/ApplyAdam	ApplyAdam'policy_network/fully_connected_2/biases,policy_network/fully_connected_2/biases/Adam.policy_network/fully_connected_2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/policy_network/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *:
_class0
.,loc:@policy_network/fully_connected_2/biases*
use_locking( *
T0*
_output_shapes
:
�
Adam/update_log_std/ApplyAdam	ApplyAdamlog_stdlog_std/Adamlog_std/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/Exp_1_grad/mul*
use_nesterov( *
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_2/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_2/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_2/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_2/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_2/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_2/biases/ApplyAdam^Adam/update_log_std/ApplyAdam^Adam/Assign^Adam/Assign_1
�
Abaseline/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB"       *
_output_shapes
:
�
?baseline/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *�)��*
_output_shapes
: 
�
?baseline/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *�)�>*
_output_shapes
: 
�
Ibaseline/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformAbaseline/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@baseline/fully_connected/weights
�
?baseline/fully_connected/weights/Initializer/random_uniform/subSub?baseline/fully_connected/weights/Initializer/random_uniform/max?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes
: 
�
?baseline/fully_connected/weights/Initializer/random_uniform/mulMulIbaseline/fully_connected/weights/Initializer/random_uniform/RandomUniform?baseline/fully_connected/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
;baseline/fully_connected/weights/Initializer/random_uniformAdd?baseline/fully_connected/weights/Initializer/random_uniform/mul?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
 baseline/fully_connected/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
�
'baseline/fully_connected/weights/AssignAssign baseline/fully_connected/weights;baseline/fully_connected/weights/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

: 
�
%baseline/fully_connected/weights/readIdentity baseline/fully_connected/weights*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
1baseline/fully_connected/biases/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
baseline/fully_connected/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
&baseline/fully_connected/biases/AssignAssignbaseline/fully_connected/biases1baseline/fully_connected/biases/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
$baseline/fully_connected/biases/readIdentitybaseline/fully_connected/biases*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
baseline/fully_connected/MatMulMatMulo%baseline/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
 baseline/fully_connected/BiasAddBiasAddbaseline/fully_connected/MatMul$baseline/fully_connected/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:��������� 
y
baseline/fully_connected/ReluRelu baseline/fully_connected/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Cbaseline/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB"        *
_output_shapes
:
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *qĜ�*
_output_shapes
: 
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *qĜ>*
_output_shapes
: 
�
Kbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformCbaseline/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:  *
dtype0*
seed2 *

seed *
T0*5
_class+
)'loc:@baseline/fully_connected_1/weights
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/subSubAbaseline/fully_connected_1/weights/Initializer/random_uniform/maxAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes
: 
�
Abaseline/fully_connected_1/weights/Initializer/random_uniform/mulMulKbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformAbaseline/fully_connected_1/weights/Initializer/random_uniform/sub*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
=baseline/fully_connected_1/weights/Initializer/random_uniformAddAbaseline/fully_connected_1/weights/Initializer/random_uniform/mulAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
"baseline/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
�
)baseline/fully_connected_1/weights/AssignAssign"baseline/fully_connected_1/weights=baseline/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:  
�
'baseline/fully_connected_1/weights/readIdentity"baseline/fully_connected_1/weights*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
3baseline/fully_connected_1/biases/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
!baseline/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
�
(baseline/fully_connected_1/biases/AssignAssign!baseline/fully_connected_1/biases3baseline/fully_connected_1/biases/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
: 
�
&baseline/fully_connected_1/biases/readIdentity!baseline/fully_connected_1/biases*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
: 
�
!baseline/fully_connected_1/MatMulMatMulbaseline/fully_connected/Relu'baseline/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:��������� 
�
"baseline/fully_connected_1/BiasAddBiasAdd!baseline/fully_connected_1/MatMul&baseline/fully_connected_1/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:��������� 
}
baseline/fully_connected_1/ReluRelu"baseline/fully_connected_1/BiasAdd*
T0*'
_output_shapes
:��������� 
�
Cbaseline/fully_connected_2/weights/Initializer/random_uniform/shapeConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB"       *
_output_shapes
:
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB
 *JQھ*
_output_shapes
: 
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/maxConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB
 *JQ�>*
_output_shapes
: 
�
Kbaseline/fully_connected_2/weights/Initializer/random_uniform/RandomUniformRandomUniformCbaseline/fully_connected_2/weights/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*5
_class+
)'loc:@baseline/fully_connected_2/weights
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/subSubAbaseline/fully_connected_2/weights/Initializer/random_uniform/maxAbaseline/fully_connected_2/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes
: 
�
Abaseline/fully_connected_2/weights/Initializer/random_uniform/mulMulKbaseline/fully_connected_2/weights/Initializer/random_uniform/RandomUniformAbaseline/fully_connected_2/weights/Initializer/random_uniform/sub*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
=baseline/fully_connected_2/weights/Initializer/random_uniformAddAbaseline/fully_connected_2/weights/Initializer/random_uniform/mulAbaseline/fully_connected_2/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
"baseline/fully_connected_2/weights
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *5
_class+
)'loc:@baseline/fully_connected_2/weights*
shared_name 
�
)baseline/fully_connected_2/weights/AssignAssign"baseline/fully_connected_2/weights=baseline/fully_connected_2/weights/Initializer/random_uniform*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
'baseline/fully_connected_2/weights/readIdentity"baseline/fully_connected_2/weights*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
3baseline/fully_connected_2/biases/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
!baseline/fully_connected_2/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_2/biases*
shared_name 
�
(baseline/fully_connected_2/biases/AssignAssign!baseline/fully_connected_2/biases3baseline/fully_connected_2/biases/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
&baseline/fully_connected_2/biases/readIdentity!baseline/fully_connected_2/biases*4
_class*
(&loc:@baseline/fully_connected_2/biases*
T0*
_output_shapes
:
�
!baseline/fully_connected_2/MatMulMatMulbaseline/fully_connected_1/Relu'baseline/fully_connected_2/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
"baseline/fully_connected_2/BiasAddBiasAdd!baseline/fully_connected_2/MatMul&baseline/fully_connected_2/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
e

baseline_1Placeholder*
dtype0*
shape:���������*#
_output_shapes
:���������
m
SqueezeSqueeze"baseline/fully_connected_2/BiasAdd*
squeeze_dims
 *
T0*
_output_shapes
:
q
$mean_squared_error/SquaredDifferenceSquaredDifference
baseline_1Squeeze*
T0*
_output_shapes
:
t
/mean_squared_error/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
dtype0*
valueB *
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
�
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:���������
�
3mean_squared_error/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference*
T0*
_output_shapes
: 
u
3mean_squared_error/assert_broadcastable/is_scalar/xConst*
dtype0*
value	B : *
_output_shapes
: 
�
1mean_squared_error/assert_broadcastable/is_scalarEqual3mean_squared_error/assert_broadcastable/is_scalar/x4mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
=mean_squared_error/assert_broadcastable/is_valid_shape/SwitchSwitch1mean_squared_error/assert_broadcastable/is_scalar1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_tIdentity?mean_squared_error/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_fIdentity=mean_squared_error/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
>mean_squared_error/assert_broadcastable/is_valid_shape/pred_idIdentity1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
?mean_squared_error/assert_broadcastable/is_valid_shape/Switch_1Switch1mean_squared_error/assert_broadcastable/is_scalar>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualdmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchfmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
dmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch3mean_squared_error/assert_broadcastable/values/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*F
_class<
:8loc:@mean_squared_error/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
�
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch4mean_squared_error/assert_broadcastable/weights/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
�
Wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityWmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimswmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch4mean_squared_error/assert_broadcastable/values/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchsmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapelmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
�
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:���������
�
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimskmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*'
_output_shapes
:���������*

Tidx0*
T0*
N
�
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
�
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch5mean_squared_error/assert_broadcastable/weights/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchumean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationnmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
set_operationa-b*
T0*<
_output_shapes*
(:���������:���������:
�
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
�
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
�
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualcmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xrmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*p
_classf
dbloc:@mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
Vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
�
<mean_squared_error/assert_broadcastable/is_valid_shape/MergeMergeVmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeAmean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
�
-mean_squared_error/assert_broadcastable/ConstConst*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
~
/mean_squared_error/assert_broadcastable/Const_1Const*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
/mean_squared_error/assert_broadcastable/Const_2Const*
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
_output_shapes
: 
}
/mean_squared_error/assert_broadcastable/Const_3Const*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
/mean_squared_error/assert_broadcastable/Const_4Const*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
z
/mean_squared_error/assert_broadcastable/Const_5Const*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
:mean_squared_error/assert_broadcastable/AssertGuard/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
<mean_squared_error/assert_broadcastable/AssertGuard/switch_tIdentity<mean_squared_error/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
<mean_squared_error/assert_broadcastable/AssertGuard/switch_fIdentity:mean_squared_error/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
;mean_squared_error/assert_broadcastable/AssertGuard/pred_idIdentity<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 

8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp=^mean_squared_error/assert_broadcastable/AssertGuard/switch_t
�
Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependencyIdentity<mean_squared_error/assert_broadcastable/AssertGuard/switch_t9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertAmean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchAmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

�
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Switch5mean_squared_error/assert_broadcastable/weights/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Switch4mean_squared_error/assert_broadcastable/values/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3Switch1mean_squared_error/assert_broadcastable/is_scalar;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Identity<mean_squared_error/assert_broadcastable/AssertGuard/switch_f;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
�
9mean_squared_error/assert_broadcastable/AssertGuard/MergeMergeHmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
�
mean_squared_error/ToFloat_3/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
X
mean_squared_error/RankRankmean_squared_error/Mul*
T0*
_output_shapes
: 
�
mean_squared_error/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
mean_squared_error/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
�
mean_squared_error/rangeRangemean_squared_error/range/startmean_squared_error/Rankmean_squared_error/range/delta*

Tidx0*#
_output_shapes
:���������
�
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
&mean_squared_error/num_present/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat_3/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
�
)mean_squared_error/num_present/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
.mean_squared_error/num_present/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
.mean_squared_error/num_present/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
�
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:���������
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
: 
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*d
_classZ
XVloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
�
umean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityumean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShape�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:���������
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*'
_output_shapes
:���������*

Tidx0*
T0*
N
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
���������*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
set_operationa-b*
T0*<
_output_shapes*
(:���������:���������:
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
�
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*�
_class�
��loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergewmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
�
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergetmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
�
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
�
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
�
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
�
dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
�
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

�
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:���������:���������
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
�
Wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergefmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:���������
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*
_output_shapes
:
�
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
~
#mean_squared_error/num_present/RankRank0mean_squared_error/num_present/broadcast_weights*
T0*
_output_shapes
: 
�
*mean_squared_error/num_present/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
�
*mean_squared_error/num_present/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
�
$mean_squared_error/num_present/rangeRange*mean_squared_error/num_present/range/start#mean_squared_error/num_present/Rank*mean_squared_error/num_present/range/delta*

Tidx0*#
_output_shapes
:���������
�
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
mean_squared_error/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
mean_squared_error/Greater/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
T0*
_output_shapes
: 
�
mean_squared_error/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
~
mean_squared_error/EqualEqualmean_squared_error/num_presentmean_squared_error/Equal/y*
T0*
_output_shapes
: 
�
"mean_squared_error/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
�
"mean_squared_error/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*
_output_shapes
: 
�
mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
�
mean_squared_error/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
�
mean_squared_error/valueSelectmean_squared_error/Greatermean_squared_error/divmean_squared_error/zeros_like*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
c
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*
_output_shapes
: 
y
4gradients_1/mean_squared_error/value_grad/zeros_likeConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
0gradients_1/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients_1/Fill4gradients_1/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
�
2gradients_1/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater4gradients_1/mean_squared_error/value_grad/zeros_likegradients_1/Fill*
T0*
_output_shapes
: 
�
:gradients_1/mean_squared_error/value_grad/tuple/group_depsNoOp1^gradients_1/mean_squared_error/value_grad/Select3^gradients_1/mean_squared_error/value_grad/Select_1
�
Bgradients_1/mean_squared_error/value_grad/tuple/control_dependencyIdentity0gradients_1/mean_squared_error/value_grad/Select;^gradients_1/mean_squared_error/value_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/mean_squared_error/value_grad/Select*
T0*
_output_shapes
: 
�
Dgradients_1/mean_squared_error/value_grad/tuple/control_dependency_1Identity2gradients_1/mean_squared_error/value_grad/Select_1;^gradients_1/mean_squared_error/value_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/mean_squared_error/value_grad/Select_1*
T0*
_output_shapes
: 
p
-gradients_1/mean_squared_error/div_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
r
/gradients_1/mean_squared_error/div_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/div_grad/Shape/gradients_1/mean_squared_error/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/gradients_1/mean_squared_error/div_grad/RealDivRealDivBgradients_1/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
�
+gradients_1/mean_squared_error/div_grad/SumSum/gradients_1/mean_squared_error/div_grad/RealDiv=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
/gradients_1/mean_squared_error/div_grad/ReshapeReshape+gradients_1/mean_squared_error/div_grad/Sum-gradients_1/mean_squared_error/div_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
m
+gradients_1/mean_squared_error/div_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
�
1gradients_1/mean_squared_error/div_grad/RealDiv_1RealDiv+gradients_1/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
�
1gradients_1/mean_squared_error/div_grad/RealDiv_2RealDiv1gradients_1/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
�
+gradients_1/mean_squared_error/div_grad/mulMulBgradients_1/mean_squared_error/value_grad/tuple/control_dependency1gradients_1/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
-gradients_1/mean_squared_error/div_grad/Sum_1Sum+gradients_1/mean_squared_error/div_grad/mul?gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
1gradients_1/mean_squared_error/div_grad/Reshape_1Reshape-gradients_1/mean_squared_error/div_grad/Sum_1/gradients_1/mean_squared_error/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
8gradients_1/mean_squared_error/div_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/div_grad/Reshape2^gradients_1/mean_squared_error/div_grad/Reshape_1
�
@gradients_1/mean_squared_error/div_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/div_grad/Reshape9^gradients_1/mean_squared_error/div_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/div_grad/Reshape*
T0*
_output_shapes
: 
�
Bgradients_1/mean_squared_error/div_grad/tuple/control_dependency_1Identity1gradients_1/mean_squared_error/div_grad/Reshape_19^gradients_1/mean_squared_error/div_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/div_grad/Reshape_1*
T0*
_output_shapes
: 
z
7gradients_1/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
1gradients_1/mean_squared_error/Sum_1_grad/ReshapeReshape@gradients_1/mean_squared_error/div_grad/tuple/control_dependency7gradients_1/mean_squared_error/Sum_1_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
: 
{
8gradients_1/mean_squared_error/Sum_1_grad/Tile/multiplesConst*
dtype0*
valueB *
_output_shapes
: 
�
.gradients_1/mean_squared_error/Sum_1_grad/TileTile1gradients_1/mean_squared_error/Sum_1_grad/Reshape8gradients_1/mean_squared_error/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
z
5gradients_1/mean_squared_error/Select_grad/zeros_likeConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
1gradients_1/mean_squared_error/Select_grad/SelectSelectmean_squared_error/EqualBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_15gradients_1/mean_squared_error/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
3gradients_1/mean_squared_error/Select_grad/Select_1Selectmean_squared_error/Equal5gradients_1/mean_squared_error/Select_grad/zeros_likeBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
;gradients_1/mean_squared_error/Select_grad/tuple/group_depsNoOp2^gradients_1/mean_squared_error/Select_grad/Select4^gradients_1/mean_squared_error/Select_grad/Select_1
�
Cgradients_1/mean_squared_error/Select_grad/tuple/control_dependencyIdentity1gradients_1/mean_squared_error/Select_grad/Select<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Select_grad/Select*
T0*
_output_shapes
: 
�
Egradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1Identity3gradients_1/mean_squared_error/Select_grad/Select_1<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*F
_class<
:8loc:@gradients_1/mean_squared_error/Select_grad/Select_1*
T0*
_output_shapes
: 
�
-gradients_1/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
out_type0*
T0*#
_output_shapes
:���������
�
,gradients_1/mean_squared_error/Sum_grad/SizeSize-gradients_1/mean_squared_error/Sum_grad/Shape*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
: 
�
+gradients_1/mean_squared_error/Sum_grad/addAddmean_squared_error/range,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
+gradients_1/mean_squared_error/Sum_grad/modFloorMod+gradients_1/mean_squared_error/Sum_grad/add,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
/gradients_1/mean_squared_error/Sum_grad/Shape_1Shape+gradients_1/mean_squared_error/Sum_grad/mod*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
:
�
3gradients_1/mean_squared_error/Sum_grad/range/startConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B : *
_output_shapes
: 
�
3gradients_1/mean_squared_error/Sum_grad/range/deltaConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
-gradients_1/mean_squared_error/Sum_grad/rangeRange3gradients_1/mean_squared_error/Sum_grad/range/start,gradients_1/mean_squared_error/Sum_grad/Size3gradients_1/mean_squared_error/Sum_grad/range/delta*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*

Tidx0*#
_output_shapes
:���������
�
2gradients_1/mean_squared_error/Sum_grad/Fill/valueConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
,gradients_1/mean_squared_error/Sum_grad/FillFill/gradients_1/mean_squared_error/Sum_grad/Shape_12gradients_1/mean_squared_error/Sum_grad/Fill/value*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
5gradients_1/mean_squared_error/Sum_grad/DynamicStitchDynamicStitch-gradients_1/mean_squared_error/Sum_grad/range+gradients_1/mean_squared_error/Sum_grad/mod-gradients_1/mean_squared_error/Sum_grad/Shape,gradients_1/mean_squared_error/Sum_grad/Fill*
N*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
1gradients_1/mean_squared_error/Sum_grad/Maximum/yConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
/gradients_1/mean_squared_error/Sum_grad/MaximumMaximum5gradients_1/mean_squared_error/Sum_grad/DynamicStitch1gradients_1/mean_squared_error/Sum_grad/Maximum/y*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
0gradients_1/mean_squared_error/Sum_grad/floordivFloorDiv-gradients_1/mean_squared_error/Sum_grad/Shape/gradients_1/mean_squared_error/Sum_grad/Maximum*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:���������
�
/gradients_1/mean_squared_error/Sum_grad/ReshapeReshape.gradients_1/mean_squared_error/Sum_1_grad/Tile5gradients_1/mean_squared_error/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
�
,gradients_1/mean_squared_error/Sum_grad/TileTile/gradients_1/mean_squared_error/Sum_grad/Reshape0gradients_1/mean_squared_error/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
�
5gradients_1/mean_squared_error/num_present_grad/ShapeShape0mean_squared_error/num_present/broadcast_weights*
out_type0*
T0*#
_output_shapes
:���������
�
4gradients_1/mean_squared_error/num_present_grad/SizeSize5gradients_1/mean_squared_error/num_present_grad/Shape*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
out_type0*
T0*
_output_shapes
: 
�
3gradients_1/mean_squared_error/num_present_grad/addAdd$mean_squared_error/num_present/range4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
3gradients_1/mean_squared_error/num_present_grad/modFloorMod3gradients_1/mean_squared_error/num_present_grad/add4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
7gradients_1/mean_squared_error/num_present_grad/Shape_1Shape3gradients_1/mean_squared_error/num_present_grad/mod*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
out_type0*
T0*
_output_shapes
:
�
;gradients_1/mean_squared_error/num_present_grad/range/startConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B : *
_output_shapes
: 
�
;gradients_1/mean_squared_error/num_present_grad/range/deltaConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
�
5gradients_1/mean_squared_error/num_present_grad/rangeRange;gradients_1/mean_squared_error/num_present_grad/range/start4gradients_1/mean_squared_error/num_present_grad/Size;gradients_1/mean_squared_error/num_present_grad/range/delta*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*

Tidx0*#
_output_shapes
:���������
�
:gradients_1/mean_squared_error/num_present_grad/Fill/valueConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
�
4gradients_1/mean_squared_error/num_present_grad/FillFill7gradients_1/mean_squared_error/num_present_grad/Shape_1:gradients_1/mean_squared_error/num_present_grad/Fill/value*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
=gradients_1/mean_squared_error/num_present_grad/DynamicStitchDynamicStitch5gradients_1/mean_squared_error/num_present_grad/range3gradients_1/mean_squared_error/num_present_grad/mod5gradients_1/mean_squared_error/num_present_grad/Shape4gradients_1/mean_squared_error/num_present_grad/Fill*
N*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
9gradients_1/mean_squared_error/num_present_grad/Maximum/yConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
�
7gradients_1/mean_squared_error/num_present_grad/MaximumMaximum=gradients_1/mean_squared_error/num_present_grad/DynamicStitch9gradients_1/mean_squared_error/num_present_grad/Maximum/y*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
8gradients_1/mean_squared_error/num_present_grad/floordivFloorDiv5gradients_1/mean_squared_error/num_present_grad/Shape7gradients_1/mean_squared_error/num_present_grad/Maximum*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:���������
�
7gradients_1/mean_squared_error/num_present_grad/ReshapeReshapeEgradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1=gradients_1/mean_squared_error/num_present_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
�
4gradients_1/mean_squared_error/num_present_grad/TileTile7gradients_1/mean_squared_error/num_present_grad/Reshape8gradients_1/mean_squared_error/num_present_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
�
-gradients_1/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:���������
r
/gradients_1/mean_squared_error/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/Mul_grad/Shape/gradients_1/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients_1/mean_squared_error/Mul_grad/mulMul,gradients_1/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
�
+gradients_1/mean_squared_error/Mul_grad/SumSum+gradients_1/mean_squared_error/Mul_grad/mul=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
/gradients_1/mean_squared_error/Mul_grad/ReshapeReshape+gradients_1/mean_squared_error/Mul_grad/Sum-gradients_1/mean_squared_error/Mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
�
-gradients_1/mean_squared_error/Mul_grad/mul_1Mul$mean_squared_error/SquaredDifference,gradients_1/mean_squared_error/Sum_grad/Tile*
T0*
_output_shapes
:
�
-gradients_1/mean_squared_error/Mul_grad/Sum_1Sum-gradients_1/mean_squared_error/Mul_grad/mul_1?gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
1gradients_1/mean_squared_error/Mul_grad/Reshape_1Reshape-gradients_1/mean_squared_error/Mul_grad/Sum_1/gradients_1/mean_squared_error/Mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
8gradients_1/mean_squared_error/Mul_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/Mul_grad/Reshape2^gradients_1/mean_squared_error/Mul_grad/Reshape_1
�
@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/Mul_grad/Reshape9^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/Mul_grad/Reshape*
T0*
_output_shapes
:
�
Bgradients_1/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity1gradients_1/mean_squared_error/Mul_grad/Reshape_19^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Mul_grad/Reshape_1*
T0*
_output_shapes
: 
�
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1Shape:mean_squared_error/num_present/broadcast_weights/ones_like*
out_type0*
T0*#
_output_shapes
:���������
�
Wgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulMul4gradients_1/mean_squared_error/num_present_grad/Tile:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
�
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumSumEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulWgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeReshapeEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Mul%mean_squared_error/num_present/Select4gradients_1/mean_squared_error/num_present_grad/Tile*
T0*
_output_shapes
:
�
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Ygradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Kgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1ReshapeGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
�
Rgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_depsNoOpJ^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeL^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1
�
Zgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeS^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape*
T0*
_output_shapes
: 
�
\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityKgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1S^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1*
T0*
_output_shapes
:
�
Pgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankRank\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/rangeRangeWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startPgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/delta*

Tidx0*#
_output_shapes
:���������
�
Ogradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/SumSum\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
;gradients_1/mean_squared_error/SquaredDifference_grad/ShapeShape
baseline_1*
out_type0*
T0*
_output_shapes
:
�
=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1ShapeSqueeze*
out_type0*
T0*#
_output_shapes
:���������
�
Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/mean_squared_error/SquaredDifference_grad/Shape=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients_1/mean_squared_error/SquaredDifference_grad/scalarConstA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
9gradients_1/mean_squared_error/SquaredDifference_grad/mulMul<gradients_1/mean_squared_error/SquaredDifference_grad/scalar@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
9gradients_1/mean_squared_error/SquaredDifference_grad/subSub
baseline_1SqueezeA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mul9gradients_1/mean_squared_error/SquaredDifference_grad/mul9gradients_1/mean_squared_error/SquaredDifference_grad/sub*
T0*
_output_shapes
:
�
9gradients_1/mean_squared_error/SquaredDifference_grad/SumSum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeReshape9gradients_1/mean_squared_error/SquaredDifference_grad/Sum;gradients_1/mean_squared_error/SquaredDifference_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
�
;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1Sum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
�
9gradients_1/mean_squared_error/SquaredDifference_grad/NegNeg?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes
:
�
Fgradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp>^gradients_1/mean_squared_error/SquaredDifference_grad/Reshape:^gradients_1/mean_squared_error/SquaredDifference_grad/Neg
�
Ngradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/mean_squared_error/SquaredDifference_grad/Reshape*
T0*#
_output_shapes
:���������
�
Pgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity9gradients_1/mean_squared_error/SquaredDifference_grad/NegG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/mean_squared_error/SquaredDifference_grad/Neg*
T0*
_output_shapes
:
�
gradients_1/Squeeze_grad/ShapeShape"baseline/fully_connected_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
 gradients_1/Squeeze_grad/ReshapeReshapePgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1gradients_1/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
?gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients_1/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:
�
Dgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients_1/Squeeze_grad/Reshape@^gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGrad
�
Lgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients_1/Squeeze_grad/ReshapeE^gradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Squeeze_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ngradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGradE^gradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/baseline/fully_connected_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
9gradients_1/baseline/fully_connected_2/MatMul_grad/MatMulMatMulLgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency'baseline/fully_connected_2/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
;gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1MatMulbaseline/fully_connected_1/ReluLgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Cgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul<^gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1
�
Kgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_2/MatMul_grad/MatMulD^gradients_1/baseline/fully_connected_2/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Mgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependency_1Identity;gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1D^gradients_1/baseline/fully_connected_2/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/baseline/fully_connected_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
9gradients_1/baseline/fully_connected_1/Relu_grad/ReluGradReluGradKgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependencybaseline/fully_connected_1/Relu*
T0*'
_output_shapes
:��������� 
�
?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/baseline/fully_connected_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
Dgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_1/Relu_grad/ReluGrad@^gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad
�
Lgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_1/Relu_grad/ReluGradE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Ngradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulMatMulLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency'baseline/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1MatMulbaseline/fully_connected/ReluLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:  
�
Cgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul<^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1
�
Kgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulD^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
Mgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1D^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
�
7gradients_1/baseline/fully_connected/Relu_grad/ReluGradReluGradKgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencybaseline/fully_connected/Relu*
T0*'
_output_shapes
:��������� 
�
=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
Bgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/Relu_grad/ReluGrad>^gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad
�
Jgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/Relu_grad/ReluGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:��������� 
�
Lgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
7gradients_1/baseline/fully_connected/MatMul_grad/MatMulMatMulJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency%baseline/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1MatMuloJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

: 
�
Agradients_1/baseline/fully_connected/MatMul_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/MatMul_grad/MatMul:^gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1
�
Igradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/MatMul_grad/MatMulB^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Kgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1Identity9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1B^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
beta1_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta1_power_1/readIdentitybeta1_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
beta2_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
beta2_power_1/readIdentitybeta2_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
7baseline/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
%baseline/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
�
,baseline/fully_connected/weights/Adam/AssignAssign%baseline/fully_connected/weights/Adam7baseline/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

: 
�
*baseline/fully_connected/weights/Adam/readIdentity%baseline/fully_connected/weights/Adam*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
9baseline/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB *    *
_output_shapes

: 
�
'baseline/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
�
.baseline/fully_connected/weights/Adam_1/AssignAssign'baseline/fully_connected/weights/Adam_19baseline/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

: 
�
,baseline/fully_connected/weights/Adam_1/readIdentity'baseline/fully_connected/weights/Adam_1*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

: 
�
6baseline/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
$baseline/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
+baseline/fully_connected/biases/Adam/AssignAssign$baseline/fully_connected/biases/Adam6baseline/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
)baseline/fully_connected/biases/Adam/readIdentity$baseline/fully_connected/biases/Adam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
8baseline/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB *    *
_output_shapes
: 
�
&baseline/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
�
-baseline/fully_connected/biases/Adam_1/AssignAssign&baseline/fully_connected/biases/Adam_18baseline/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
�
+baseline/fully_connected/biases/Adam_1/readIdentity&baseline/fully_connected/biases/Adam_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
9baseline/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
'baseline/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
�
.baseline/fully_connected_1/weights/Adam/AssignAssign'baseline/fully_connected_1/weights/Adam9baseline/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:  
�
,baseline/fully_connected_1/weights/Adam/readIdentity'baseline/fully_connected_1/weights/Adam*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
;baseline/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB  *    *
_output_shapes

:  
�
)baseline/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:  *
dtype0*
shape
:  *5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
�
0baseline/fully_connected_1/weights/Adam_1/AssignAssign)baseline/fully_connected_1/weights/Adam_1;baseline/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:  
�
.baseline/fully_connected_1/weights/Adam_1/readIdentity)baseline/fully_connected_1/weights/Adam_1*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:  
�
8baseline/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
&baseline/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
�
-baseline/fully_connected_1/biases/Adam/AssignAssign&baseline/fully_connected_1/biases/Adam8baseline/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
: 
�
+baseline/fully_connected_1/biases/Adam/readIdentity&baseline/fully_connected_1/biases/Adam*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
: 
�
:baseline/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB *    *
_output_shapes
: 
�
(baseline/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
�
/baseline/fully_connected_1/biases/Adam_1/AssignAssign(baseline/fully_connected_1/biases/Adam_1:baseline/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
: 
�
-baseline/fully_connected_1/biases/Adam_1/readIdentity(baseline/fully_connected_1/biases/Adam_1*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
: 
�
9baseline/fully_connected_2/weights/Adam/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
'baseline/fully_connected_2/weights/Adam
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *5
_class+
)'loc:@baseline/fully_connected_2/weights*
shared_name 
�
.baseline/fully_connected_2/weights/Adam/AssignAssign'baseline/fully_connected_2/weights/Adam9baseline/fully_connected_2/weights/Adam/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
,baseline/fully_connected_2/weights/Adam/readIdentity'baseline/fully_connected_2/weights/Adam*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
;baseline/fully_connected_2/weights/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_2/weights*
valueB *    *
_output_shapes

: 
�
)baseline/fully_connected_2/weights/Adam_1
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *5
_class+
)'loc:@baseline/fully_connected_2/weights*
shared_name 
�
0baseline/fully_connected_2/weights/Adam_1/AssignAssign)baseline/fully_connected_2/weights/Adam_1;baseline/fully_connected_2/weights/Adam_1/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking(*
T0*
_output_shapes

: 
�
.baseline/fully_connected_2/weights/Adam_1/readIdentity)baseline/fully_connected_2/weights/Adam_1*5
_class+
)'loc:@baseline/fully_connected_2/weights*
T0*
_output_shapes

: 
�
8baseline/fully_connected_2/biases/Adam/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
&baseline/fully_connected_2/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_2/biases*
shared_name 
�
-baseline/fully_connected_2/biases/Adam/AssignAssign&baseline/fully_connected_2/biases/Adam8baseline/fully_connected_2/biases/Adam/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
+baseline/fully_connected_2/biases/Adam/readIdentity&baseline/fully_connected_2/biases/Adam*4
_class*
(&loc:@baseline/fully_connected_2/biases*
T0*
_output_shapes
:
�
:baseline/fully_connected_2/biases/Adam_1/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_2/biases*
valueB*    *
_output_shapes
:
�
(baseline/fully_connected_2/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_2/biases*
shared_name 
�
/baseline/fully_connected_2/biases/Adam_1/AssignAssign(baseline/fully_connected_2/biases/Adam_1:baseline/fully_connected_2/biases/Adam_1/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking(*
T0*
_output_shapes
:
�
-baseline/fully_connected_2/biases/Adam_1/readIdentity(baseline/fully_connected_2/biases/Adam_1*4
_class*
(&loc:@baseline/fully_connected_2/biases*
T0*
_output_shapes
:
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *���<*
_output_shapes
: 
Q
Adam_1/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Q
Adam_1/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
8Adam_1/update_baseline/fully_connected/weights/ApplyAdam	ApplyAdam baseline/fully_connected/weights%baseline/fully_connected/weights/Adam'baseline/fully_connected/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonKgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking( *
T0*
_output_shapes

: 
�
7Adam_1/update_baseline/fully_connected/biases/ApplyAdam	ApplyAdambaseline/fully_connected/biases$baseline/fully_connected/biases/Adam&baseline/fully_connected/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonLgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
:Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam	ApplyAdam"baseline/fully_connected_1/weights'baseline/fully_connected_1/weights/Adam)baseline/fully_connected_1/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonMgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:  
�
9Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam	ApplyAdam!baseline/fully_connected_1/biases&baseline/fully_connected_1/biases/Adam(baseline/fully_connected_1/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
: 
�
:Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam	ApplyAdam"baseline/fully_connected_2/weights'baseline/fully_connected_2/weights/Adam)baseline/fully_connected_2/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonMgradients_1/baseline/fully_connected_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *5
_class+
)'loc:@baseline/fully_connected_2/weights*
use_locking( *
T0*
_output_shapes

: 
�
9Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam	ApplyAdam!baseline/fully_connected_2/biases&baseline/fully_connected_2/biases/Adam(baseline/fully_connected_2/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/baseline/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *4
_class*
(&loc:@baseline/fully_connected_2/biases*
use_locking( *
T0*
_output_shapes
:
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta19^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta29^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
�
Adam_1NoOp9^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_2/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_2/biases/ApplyAdam^Adam_1/Assign^Adam_1/Assign_1
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
: ""
train_op

Adam
Adam_1"�
trainable_variables��
�
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
�
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
�
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
�
*policy_network/fully_connected_2/weights:0/policy_network/fully_connected_2/weights/Assign/policy_network/fully_connected_2/weights/read:02Epolicy_network/fully_connected_2/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_2/biases:0.policy_network/fully_connected_2/biases/Assign.policy_network/fully_connected_2/biases/read:02;policy_network/fully_connected_2/biases/Initializer/zeros:0
Q
	log_std:0log_std/Assignlog_std/read:02$log_std/Initializer/random_uniform:0
�
"baseline/fully_connected/weights:0'baseline/fully_connected/weights/Assign'baseline/fully_connected/weights/read:02=baseline/fully_connected/weights/Initializer/random_uniform:0
�
!baseline/fully_connected/biases:0&baseline/fully_connected/biases/Assign&baseline/fully_connected/biases/read:023baseline/fully_connected/biases/Initializer/zeros:0
�
$baseline/fully_connected_1/weights:0)baseline/fully_connected_1/weights/Assign)baseline/fully_connected_1/weights/read:02?baseline/fully_connected_1/weights/Initializer/random_uniform:0
�
#baseline/fully_connected_1/biases:0(baseline/fully_connected_1/biases/Assign(baseline/fully_connected_1/biases/read:025baseline/fully_connected_1/biases/Initializer/zeros:0
�
$baseline/fully_connected_2/weights:0)baseline/fully_connected_2/weights/Assign)baseline/fully_connected_2/weights/read:02?baseline/fully_connected_2/weights/Initializer/random_uniform:0
�
#baseline/fully_connected_2/biases:0(baseline/fully_connected_2/biases/Assign(baseline/fully_connected_2/biases/read:025baseline/fully_connected_2/biases/Initializer/zeros:0"�?
	variables�?�?
�
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
�
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
�
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
�
*policy_network/fully_connected_2/weights:0/policy_network/fully_connected_2/weights/Assign/policy_network/fully_connected_2/weights/read:02Epolicy_network/fully_connected_2/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_2/biases:0.policy_network/fully_connected_2/biases/Assign.policy_network/fully_connected_2/biases/read:02;policy_network/fully_connected_2/biases/Initializer/zeros:0
Q
	log_std:0log_std/Assignlog_std/read:02$log_std/Initializer/random_uniform:0
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
0policy_network/fully_connected_1/biases/Adam_1:05policy_network/fully_connected_1/biases/Adam_1/Assign5policy_network/fully_connected_1/biases/Adam_1/read:02Bpolicy_network/fully_connected_1/biases/Adam_1/Initializer/zeros:0
�
/policy_network/fully_connected_2/weights/Adam:04policy_network/fully_connected_2/weights/Adam/Assign4policy_network/fully_connected_2/weights/Adam/read:02Apolicy_network/fully_connected_2/weights/Adam/Initializer/zeros:0
�
1policy_network/fully_connected_2/weights/Adam_1:06policy_network/fully_connected_2/weights/Adam_1/Assign6policy_network/fully_connected_2/weights/Adam_1/read:02Cpolicy_network/fully_connected_2/weights/Adam_1/Initializer/zeros:0
�
.policy_network/fully_connected_2/biases/Adam:03policy_network/fully_connected_2/biases/Adam/Assign3policy_network/fully_connected_2/biases/Adam/read:02@policy_network/fully_connected_2/biases/Adam/Initializer/zeros:0
�
0policy_network/fully_connected_2/biases/Adam_1:05policy_network/fully_connected_2/biases/Adam_1/Assign5policy_network/fully_connected_2/biases/Adam_1/read:02Bpolicy_network/fully_connected_2/biases/Adam_1/Initializer/zeros:0
\
log_std/Adam:0log_std/Adam/Assignlog_std/Adam/read:02 log_std/Adam/Initializer/zeros:0
d
log_std/Adam_1:0log_std/Adam_1/Assignlog_std/Adam_1/read:02"log_std/Adam_1/Initializer/zeros:0
�
"baseline/fully_connected/weights:0'baseline/fully_connected/weights/Assign'baseline/fully_connected/weights/read:02=baseline/fully_connected/weights/Initializer/random_uniform:0
�
!baseline/fully_connected/biases:0&baseline/fully_connected/biases/Assign&baseline/fully_connected/biases/read:023baseline/fully_connected/biases/Initializer/zeros:0
�
$baseline/fully_connected_1/weights:0)baseline/fully_connected_1/weights/Assign)baseline/fully_connected_1/weights/read:02?baseline/fully_connected_1/weights/Initializer/random_uniform:0
�
#baseline/fully_connected_1/biases:0(baseline/fully_connected_1/biases/Assign(baseline/fully_connected_1/biases/read:025baseline/fully_connected_1/biases/Initializer/zeros:0
�
$baseline/fully_connected_2/weights:0)baseline/fully_connected_2/weights/Assign)baseline/fully_connected_2/weights/read:02?baseline/fully_connected_2/weights/Initializer/random_uniform:0
�
#baseline/fully_connected_2/biases:0(baseline/fully_connected_2/biases/Assign(baseline/fully_connected_2/biases/read:025baseline/fully_connected_2/biases/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
�
'baseline/fully_connected/weights/Adam:0,baseline/fully_connected/weights/Adam/Assign,baseline/fully_connected/weights/Adam/read:029baseline/fully_connected/weights/Adam/Initializer/zeros:0
�
)baseline/fully_connected/weights/Adam_1:0.baseline/fully_connected/weights/Adam_1/Assign.baseline/fully_connected/weights/Adam_1/read:02;baseline/fully_connected/weights/Adam_1/Initializer/zeros:0
�
&baseline/fully_connected/biases/Adam:0+baseline/fully_connected/biases/Adam/Assign+baseline/fully_connected/biases/Adam/read:028baseline/fully_connected/biases/Adam/Initializer/zeros:0
�
(baseline/fully_connected/biases/Adam_1:0-baseline/fully_connected/biases/Adam_1/Assign-baseline/fully_connected/biases/Adam_1/read:02:baseline/fully_connected/biases/Adam_1/Initializer/zeros:0
�
)baseline/fully_connected_1/weights/Adam:0.baseline/fully_connected_1/weights/Adam/Assign.baseline/fully_connected_1/weights/Adam/read:02;baseline/fully_connected_1/weights/Adam/Initializer/zeros:0
�
+baseline/fully_connected_1/weights/Adam_1:00baseline/fully_connected_1/weights/Adam_1/Assign0baseline/fully_connected_1/weights/Adam_1/read:02=baseline/fully_connected_1/weights/Adam_1/Initializer/zeros:0
�
(baseline/fully_connected_1/biases/Adam:0-baseline/fully_connected_1/biases/Adam/Assign-baseline/fully_connected_1/biases/Adam/read:02:baseline/fully_connected_1/biases/Adam/Initializer/zeros:0
�
*baseline/fully_connected_1/biases/Adam_1:0/baseline/fully_connected_1/biases/Adam_1/Assign/baseline/fully_connected_1/biases/Adam_1/read:02<baseline/fully_connected_1/biases/Adam_1/Initializer/zeros:0
�
)baseline/fully_connected_2/weights/Adam:0.baseline/fully_connected_2/weights/Adam/Assign.baseline/fully_connected_2/weights/Adam/read:02;baseline/fully_connected_2/weights/Adam/Initializer/zeros:0
�
+baseline/fully_connected_2/weights/Adam_1:00baseline/fully_connected_2/weights/Adam_1/Assign0baseline/fully_connected_2/weights/Adam_1/read:02=baseline/fully_connected_2/weights/Adam_1/Initializer/zeros:0
�
(baseline/fully_connected_2/biases/Adam:0-baseline/fully_connected_2/biases/Adam/Assign-baseline/fully_connected_2/biases/Adam/read:02:baseline/fully_connected_2/biases/Adam/Initializer/zeros:0
�
*baseline/fully_connected_2/biases/Adam_1:0/baseline/fully_connected_2/biases/Adam_1/Assign/baseline/fully_connected_2/biases/Adam_1/read:02<baseline/fully_connected_2/biases/Adam_1/Initializer/zeros:0"(
losses

mean_squared_error/value:0"��
cond_context۱ױ
�
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textZmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *�
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
omean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
jmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
emean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
6mean_squared_error/assert_broadcastable/values/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0�
6mean_squared_error/assert_broadcastable/values/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�
7mean_squared_error/assert_broadcastable/weights/shape:0ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�
\mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*�
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
�
@mean_squared_error/assert_broadcastable/is_valid_shape/cond_text@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0 *�
3mean_squared_error/assert_broadcastable/is_scalar:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0x
3mean_squared_error/assert_broadcastable/is_scalar:0Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
�
Bmean_squared_error/assert_broadcastable/is_valid_shape/cond_text_1@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0*�
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0
5mean_squared_error/assert_broadcastable/values/rank:0
6mean_squared_error/assert_broadcastable/values/shape:0
6mean_squared_error/assert_broadcastable/weights/rank:0
7mean_squared_error/assert_broadcastable/weights/shape:0�
5mean_squared_error/assert_broadcastable/values/rank:0fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0�
6mean_squared_error/assert_broadcastable/values/shape:0umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
7mean_squared_error/assert_broadcastable/weights/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
6mean_squared_error/assert_broadcastable/weights/rank:0hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
�
=mean_squared_error/assert_broadcastable/AssertGuard/cond_text=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0 *�
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0
�
?mean_squared_error/assert_broadcastable/AssertGuard/cond_text_1=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0*�
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:0
Jmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0
3mean_squared_error/assert_broadcastable/is_scalar:0
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0
6mean_squared_error/assert_broadcastable/values/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0�
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0|
3mean_squared_error/assert_broadcastable/is_scalar:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
6mean_squared_error/assert_broadcastable/values/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0�
7mean_squared_error/assert_broadcastable/weights/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0
�
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textxmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *�
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0�
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�	
zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*�
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
�
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
�
`mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*�
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0�
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0�
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0�mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
�
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *�
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0
�
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0�
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0�
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0�
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0"H
	summaries;
9
Avg_Reward:0
Max_Reward:0
Std_Reward:0
Eval_Reward:0"�
model_variables��
�
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
�
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
�
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
�
*policy_network/fully_connected_2/weights:0/policy_network/fully_connected_2/weights/Assign/policy_network/fully_connected_2/weights/read:02Epolicy_network/fully_connected_2/weights/Initializer/random_uniform:0
�
)policy_network/fully_connected_2/biases:0.policy_network/fully_connected_2/biases/Assign.policy_network/fully_connected_2/biases/read:02;policy_network/fully_connected_2/biases/Initializer/zeros:0
�
"baseline/fully_connected/weights:0'baseline/fully_connected/weights/Assign'baseline/fully_connected/weights/read:02=baseline/fully_connected/weights/Initializer/random_uniform:0
�
!baseline/fully_connected/biases:0&baseline/fully_connected/biases/Assign&baseline/fully_connected/biases/read:023baseline/fully_connected/biases/Initializer/zeros:0
�
$baseline/fully_connected_1/weights:0)baseline/fully_connected_1/weights/Assign)baseline/fully_connected_1/weights/read:02?baseline/fully_connected_1/weights/Initializer/random_uniform:0
�
#baseline/fully_connected_1/biases:0(baseline/fully_connected_1/biases/Assign(baseline/fully_connected_1/biases/read:025baseline/fully_connected_1/biases/Initializer/zeros:0
�
$baseline/fully_connected_2/weights:0)baseline/fully_connected_2/weights/Assign)baseline/fully_connected_2/weights/read:02?baseline/fully_connected_2/weights/Initializer/random_uniform:0
�
#baseline/fully_connected_2/biases:0(baseline/fully_connected_2/biases/Assign(baseline/fully_connected_2/biases/read:025baseline/fully_connected_2/biases/Initializer/zeros:0���1X       ��/�	Ω�Ң�A*M


Avg_Reward�&��


Max_Reward��


Std_Reward���A

Eval_Reward�V���:Z       o��	�}�Ң�A*M


Avg_Reward�w��


Max_Reward���


Std_RewardD�A

Eval_Reward����B��Z       o��	5*��Ң�A*M


Avg_Reward���


Max_Reward%���


Std_Reward���A

Eval_Reward3��ģkB�Z       o��	����Ң�A*M


Avg_Reward�p��


Max_Reward����


Std_Rewardli�A

Eval_Reward�7�ĳ�Z       o��	xk�Ң�A*M


Avg_Reward�$��


Max_Reward�Z��


Std_RewardSj_A

Eval_Reward��Ē/$zZ       o��	"���Ң�A*M


Avg_RewardϨk�


Max_Reward��P�


Std_Reward��+A

Eval_Reward��uĐ_dZ       o��	Z-QӢ�A*M


Avg_RewardfT<�


Max_Reward���


Std_Reward%vA

Eval_RewardF[\�1tfZ       o��	|}Ӣ�A*M


Avg_Reward�"�


Max_Reward$��


Std_Reward钏A

Eval_Rewardq*��F�"Z       o��	M��Ӣ�A*M


Avg_Reward&�


Max_Reward��


Std_Reward ݢA

Eval_Reward���ܐZ       o��	鶨+Ӣ�A	*M


Avg_Rewardu��


Max_Reward���


Std_Reward��YA

Eval_Reward�8Ĕ���Z       o��	=n9Ӣ�A
*M


Avg_Reward���


Max_Reward��


Std_RewardOUNA

Eval_Reward��Ŀ�>�Z       o��	}]�GӢ�A*M


Avg_Reward���


Max_Reward5���


Std_Reward�xA

Eval_Rewardߒ��StJqZ       o��	�@UӢ�A*M


Avg_Rewardz���


Max_Reward�7��


Std_Reward,�2A

Eval_Reward����U9f$Z       o��	��\bӢ�A*M


Avg_Reward.d��


Max_RewardJ`��


Std_Reward&A

Eval_Reward3��
�2�Z       o��	�1�oӢ�A*M


Avg_Reward���


Max_RewardY��


Std_Reward�gA

Eval_RewardDD��VW*TZ       o��	�}Ӣ�A*M


Avg_Reward�=��


Max_Reward S�


Std_Rewardy�A

Eval_Reward_��Ö�(�Z       o��	'�^�Ӣ�A*M


Avg_Reward�Ә�


Max_RewardY�l�


Std_Reward0"�@

Eval_Reward칪� ��^Z       o��	�2�Ӣ�A*M


Avg_Reward��


Max_Rewardx�M�


Std_Reward*�A

Eval_Reward�Y�`o��Z       o��	c�l�Ӣ�A*M


Avg_Rewardd���


Max_Reward�D�


Std_Reward��A

Eval_Reward뤣�
���Z       o��	py��Ӣ�A*M


Avg_Rewardj�r�


Max_Reward�oD�


Std_RewardMO�@

Eval_Reward:�p�t�;�Z       o��	_�̾Ӣ�A*M


Avg_Rewardml�


Max_Rewardz|&�


Std_Reward�A

Eval_Reward���LC`�Z       o��	�x�Ӣ�A*M


Avg_Reward��W�


Max_Reward��


Std_Reward���@

Eval_RewardB�H�H9I�Z       o��	��Y�Ӣ�A*M


Avg_Reward��<�


Max_Reward�	�


Std_RewardB��@

Eval_Reward��QÖ���Z       o��	AJ�Ӣ�A*M


Avg_Rewardn�?�


Max_Reward�D	�


Std_Rewardƭ�@

Eval_Reward�n�T)��Z       o��	����Ӣ�A*M


Avg_Reward�u+�


Max_Reward����


Std_Reward$��@

Eval_Reward}:`�hu��Z       o��	�5oԢ�A*M


Avg_Reward��


Max_Reward�;��


Std_Reward�4�@

Eval_Reward��%ÅI�Z       o��	&85Ԣ�A*M


Avg_Reward��


Max_Reward]ɏ�


Std_Reward���@

Eval_RewardÀd��Z       o��	��Ԣ�A*M


Avg_Rewards���


Max_Reward(�t�


Std_Reward�T�@

Eval_Reward>δ�s�Z       o��	�{�*Ԣ�A*M


Avg_Reward{���


Max_Reward���


Std_Reward�@�@

Eval_Reward4���,ƫ�Z       o��	��7Ԣ�A*M


Avg_Reward�q��


Max_Reward_Yl�


Std_Reward���@

Eval_Reward_Yl��G�>Z       o��	=�AEԢ�A*M


Avg_Reward�Q��


Max_Reward�	�


Std_RewardX��@

Eval_Reward;���8�7�Z       o��	�<�RԢ�A*M


Avg_Reward�V��


Max_Reward4���


Std_Reward-�A

Eval_Reward�o�¢�+9Z       o��	��b`Ԣ�A *M


Avg_Reward�ӯ�


Max_Reward���?


Std_Reward��A

Eval_Reward��q��*��Z       o��	z40nԢ�A!*M


Avg_Reward�S��


Max_Reward����


Std_Reward��@

Eval_Reward0ۑ�� "�Z       o��	�W
|Ԣ�A"*M


Avg_Rewardk{��


Max_Reward���?


Std_Reward1� A

Eval_Reward���H%H&Z       o��	�쯉Ԣ�A#*M


Avg_Rewardwp��


Max_Reward��l�


Std_RewardM�A

Eval_RewardV�"Æ,2Z       o��	K�g�Ԣ�A$*M


Avg_Reward,Ӿ�


Max_RewardA�HA


Std_Reward�|'A

Eval_Rewardȱ;@{lŇZ       o��	iVS�Ԣ�A%*M


Avg_Reward�+��


Max_Rewardo]�


Std_Rewardq�A

Eval_Reward����l�Z       o��	@�Ԣ�A&*M


Avg_Reward`��


Max_Rewarde�aB


Std_Reward��=A

Eval_Rewardz.� ��Z       o��	�)�Ԣ�A'*M


Avg_Reward=��


Max_Reward�$�A


Std_Reward�.,A

Eval_Reward��������Z       o��	��Ԣ�A(*M


Avg_RewardP���


Max_Reward��A


Std_Reward1V#A

Eval_Reward��A���6Z       o��	vlG�Ԣ�A)*M


Avg_Reward)���


Max_Reward&� B


Std_Reward��5A

Eval_Reward�D�9q��Z       o��	곞�Ԣ�A**M


Avg_Reward�I��


Max_Reward�VB


Std_Reward��$A

Eval_Reward�����?�Z       o��	h�I�Ԣ�A+*M


Avg_Reward�]X�


Max_RewardQ�B


Std_Reward|WA

Eval_Reward��Sm�Z       o��	 �� բ�A,*M


Avg_Reward��~�


Max_RewardE?(B


Std_Reward@�3A

Eval_Reward44��8hS�Z       o��	�~�բ�A-*M


Avg_Reward�;N�


Max_Reward��B


Std_Reward���@

Eval_Reward6k�����Z       o��	�&Aբ�A.*M


Avg_Reward;��


Max_Reward@ޑ�


Std_Reward���@

Eval_Reward�=µ���Z       o��	�2(բ�A/*M


Avg_Reward^.��


Max_Reward� �


Std_Reward�Z�@

Eval_Reward�\k�Z�~Z       o��	��4բ�A0*M


Avg_Reward/�R�


Max_RewardY���


Std_Reward��@

Eval_Rewardeً�##DZ       o��	�,Aբ�A1*M


Avg_Rewardbl�


Max_Reward��A


Std_Rewardʵ�@

Eval_Reward�½�}5Z       o��	q�Nբ�A2*M


Avg_RewardO�9�


Max_RewardB


Std_Reward���@

Eval_Reward��F�<�tuZ       o��	h��Zբ�A3*M


Avg_Rewardi��


Max_RewardzA


Std_Reward��@

Eval_Reward��Q��;ČZ       o��	��`hբ�A4*M


Avg_Reward����


Max_Reward-��A


Std_Reward.�@

Eval_Rewardp�-��]7�Z       o��	'jEuբ�A5*M


Avg_RewardB�ξ


Max_Reward��B


Std_Reward^ȟ@

Eval_Reward�:�-��Z       o��	|m�բ�A6*M


Avg_Rewardq�A


Max_Reward睎B


Std_Reward��@

Eval_Reward,rA���Z       o��	�f�բ�A7*M


Avg_Reward��B


Max_Reward�̩B


Std_Reward�j�@

Eval_Reward��B�Z       o��	�u�բ�A8*M


Avg_Reward9��B


Max_Reward��C


Std_Reward���@

Eval_Reward�ϭB�{Z       o��	B覬բ�A9*M


Avg_Reward�B


Max_Reward{�C


Std_Rewardb��@

Eval_RewardP��A&��Z       o��	��բ�A:*M


Avg_Reward��C


Max_RewardǜDC


Std_Reward���@

Eval_Reward��9C�f��Z       o��	�Ō�բ�A;*M


Avg_Reward�EC


Max_Reward�Y�C


Std_RewardBA

Eval_Rewardp SC�HIZ       o��	?q3�բ�A<*M


Avg_RewardFӄC


Max_Reward�׾C


Std_Reward:�A

Eval_Reward��@CU��Z       o��	��5�բ�A=*M


Avg_Reward��C


Max_Reward���C


Std_Reward�AUA

Eval_Reward��CsB� Z       o��	ĕ��բ�A>*M


Avg_Reward+ �C


Max_Reward�'D


Std_RewardIX�A

Eval_Reward�*�C摴*Z       o��	�Ͽ ֢�A?*M


Avg_RewardVy�C


Max_Reward]��C


Std_Reward� �A

Eval_RewardV��C7�Z       o��	��`֢�A@*M


Avg_Reward7�iC


Max_Reward���C


Std_Reward���A

Eval_RewardW��C��E>Z       o��	E,�֢�AA*M


Avg_Reward��C


Max_Reward��C


Std_Reward�_B

Eval_Reward)-�C�>��Z       o��	e�+֢�AB*M


Avg_Reward��xC


Max_Reward�*�C


Std_Rewardo,B

Eval_RewardG.�C��Y�Z       o��	�\�8֢�AC*M


Avg_Reward�J�C


Max_RewardAq"D


Std_Reward�d\B

Eval_Reward�`D1��Z       o��	���F֢�AD*M


Avg_Reward ��C


Max_RewardN�'D


Std_RewardH�B

Eval_Reward�VD>�MZ       o��	\�~T֢�AE*M


Avg_Reward���C


Max_Reward�"D


Std_Reward�8B

Eval_Reward?*�C��Z       o��	�Bb֢�AF*M


Avg_Reward�C


Max_RewardG�.D


Std_Reward�B

Eval_Reward
�i�33��Z       o��	�,,p֢�AG*M


Avg_RewardQT_C


Max_RewardU<!D


Std_Reward�ڀB

Eval_RewardKF���Z       o��	F��}֢�AH*M


Avg_Rewardg�@B


Max_Rewardd_�C


Std_Rewardk B

Eval_Reward�^9���Z       o��	��֢�AI*M


Avg_RewardAÙA


Max_Reward�(�C


Std_Reward#��A

Eval_Reward9Wv�ZWv,Z       o��	�E��֢�AJ*M


Avg_Reward��A


Max_Reward8t�C


Std_Reward�,�A

Eval_RewardN�������Z       o��	t@N�֢�AK*M


Avg_RewardC��


Max_Reward��rC


Std_Reward�r�A

Eval_Reward�֎6�{Z       o��	Y�֢�AL*M


Avg_Reward�BB


Max_RewardņC


Std_Reward1�A

Eval_Reward�uC5�+VZ       o��	��֢�AM*M


Avg_Reward@3C


Max_Reward�~�C


Std_Reward]�A

Eval_Reward��C��Z       o��	���֢�AN*M


Avg_Reward�+�C


Max_Reward���C


Std_Reward�<DA

Eval_Reward;�C���Z       o��	�y�֢�AO*M


Avg_Rewardf��C


Max_Reward�`�C


Std_Reward�B

Eval_Reward�ָCcF	Z       o��	��}�֢�AP*M


Avg_Reward��\C


Max_Reward��D


Std_Reward��bB

Eval_Reward��.@ʥ�Z       o��	����֢�AQ*M


Avg_Reward�VC


Max_Reward��(D


Std_RewardH�{B

Eval_Reward�q����ͯZ       o��	�ע�AR*M


Avg_Reward��7C


Max_Reward�%D


Std_Rewardֵ�B

Eval_Reward�d���Z       o��	��ע�AS*M


Avg_Reward9h�B


Max_Reward/uD


Std_Reward��3B

Eval_Rewardo���FZ       o��	��  ע�AT*M


Avg_Rewardd�C


Max_RewardƪC


Std_RewardE�B

Eval_RewardP�´���Z       o��	�H-.ע�AU*M


Avg_Reward?	1C


Max_Rewarde]�C


Std_Reward��A

Eval_RewardeW�C�+Z       o��	L�\<ע�AV*M


Avg_Reward:6qC


Max_Reward��C


Std_Reward˫,B

Eval_Reward1C�W��Z       o��	#=Kע�AW*M


Avg_RewardƖ�C


Max_Reward�4,D


Std_Reward�xuB

Eval_Reward7YKwZ       o��	"5MYע�AX*M


Avg_Rewardն�C


Max_RewardAjDD


Std_Reward<�B

Eval_Reward]�%D
LZ       o��	��gע�AY*M


Avg_RewardySD


Max_Reward�HD


Std_RewardܳnB

Eval_Reward�W@D�R�"Z       o��	Hovuע�AZ*M


Avg_Reward�V3D


Max_RewardHLSD


Std_Reward9�A

Eval_Reward5�D�� Z       o��	�o�ע�A[*M


Avg_Rewards�(D


Max_RewardНDD


Std_Reward�IA

Eval_Reward-�%D6k*�Z       o��	,��ע�A\*M


Avg_Reward�4D


Max_Reward��cD


Std_Reward{�B

Eval_Reward�>D/9�Z       o��	��9�ע�A]*M


Avg_RewardfCTD


Max_Rewardw^tD


Std_RewardB�B

Eval_Reward7B:"Z       o��	g��ע�A^*M


Avg_RewardM�ND


Max_Reward�f{D


Std_Reward1AnB

Eval_Reward��tDDC3Z       o��	h"��ע�A_*M


Avg_Reward;�:D


Max_Reward7MiD


Std_Reward�VB

Eval_Reward�K\D�� �Z       o��	o���ע�A`*M


Avg_Reward�;D


Max_Reward��VD


Std_Reward�P�B

Eval_Reward=�ND����Z       o��	�e/�ע�Aa*M


Avg_Reward&JD


Max_Reward�+kD


Std_Reward��B

Eval_Reward�	iDG�gZ       o��	ø��ע�Ab*M


Avg_RewardT�dD


Max_RewardS<�D


Std_Rewardl��A

Eval_Reward��`D��Z       o��	��t�ע�Ac*M


Avg_Reward�BTD


Max_Reward�NkD


Std_Reward�VuA

Eval_Reward�dD�D�