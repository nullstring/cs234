       £K"	  Аг@§÷Abrain.Event:2ТоX!эD     4ј§Д	?T¶г@§÷A"рЙ
d
oPlaceholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
{
a_contPlaceholder*
dtype0*%
shape:€€€€€€€€€€€€€€€€€€*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
`
advtgPlaceholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
”
Gpolicy_network/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB"      *
_output_shapes
:
≈
Epolicy_network/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *М7њ*
_output_shapes
: 
≈
Epolicy_network/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *М7?*
_output_shapes
: 
≥
Opolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformGpolicy_network/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*9
_class/
-+loc:@policy_network/fully_connected/weights
ґ
Epolicy_network/fully_connected/weights/Initializer/random_uniform/subSubEpolicy_network/fully_connected/weights/Initializer/random_uniform/maxEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes
: 
»
Epolicy_network/fully_connected/weights/Initializer/random_uniform/mulMulOpolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformEpolicy_network/fully_connected/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
Ї
Apolicy_network/fully_connected/weights/Initializer/random_uniformAddEpolicy_network/fully_connected/weights/Initializer/random_uniform/mulEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
’
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
ѓ
-policy_network/fully_connected/weights/AssignAssign&policy_network/fully_connected/weightsApolicy_network/fully_connected/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
√
+policy_network/fully_connected/weights/readIdentity&policy_network/fully_connected/weights*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
Њ
7policy_network/fully_connected/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
Ћ
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
Ю
,policy_network/fully_connected/biases/AssignAssign%policy_network/fully_connected/biases7policy_network/fully_connected/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
Љ
*policy_network/fully_connected/biases/readIdentity%policy_network/fully_connected/biases*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
Ј
%policy_network/fully_connected/MatMulMatMulo+policy_network/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
Ќ
&policy_network/fully_connected/BiasAddBiasAdd%policy_network/fully_connected/MatMul*policy_network/fully_connected/biases/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
Е
#policy_network/fully_connected/ReluRelu&policy_network/fully_connected/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
„
Ipolicy_network/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB"      *
_output_shapes
:
…
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *0њ*
_output_shapes
: 
…
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *0?*
_output_shapes
: 
є
Qpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*;
_class1
/-loc:@policy_network/fully_connected_1/weights
Њ
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/subSubGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes
: 
–
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulMulQpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
¬
Cpolicy_network/fully_connected_1/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
ў
(policy_network/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
Ј
/policy_network/fully_connected_1/weights/AssignAssign(policy_network/fully_connected_1/weightsCpolicy_network/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
…
-policy_network/fully_connected_1/weights/readIdentity(policy_network/fully_connected_1/weights*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
¬
9policy_network/fully_connected_1/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
ѕ
'policy_network/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
¶
.policy_network/fully_connected_1/biases/AssignAssign'policy_network/fully_connected_1/biases9policy_network/fully_connected_1/biases/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
¬
,policy_network/fully_connected_1/biases/readIdentity'policy_network/fully_connected_1/biases*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
Ё
'policy_network/fully_connected_1/MatMulMatMul#policy_network/fully_connected/Relu-policy_network/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
”
(policy_network/fully_connected_1/BiasAddBiasAdd'policy_network/fully_connected_1/MatMul,policy_network/fully_connected_1/biases/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
О
(log_std/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@log_std*
valueB:*
_output_shapes
:
З
&log_std/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@log_std*
valueB
 *„≥Ёњ*
_output_shapes
: 
З
&log_std/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@log_std*
valueB
 *„≥Ё?*
_output_shapes
: 
“
0log_std/Initializer/random_uniform/RandomUniformRandomUniform(log_std/Initializer/random_uniform/shape*
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*
_class
loc:@log_std
Ї
&log_std/Initializer/random_uniform/subSub&log_std/Initializer/random_uniform/max&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
: 
»
&log_std/Initializer/random_uniform/mulMul0log_std/Initializer/random_uniform/RandomUniform&log_std/Initializer/random_uniform/sub*
_class
loc:@log_std*
T0*
_output_shapes
:
Ї
"log_std/Initializer/random_uniformAdd&log_std/Initializer/random_uniform/mul&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
:
П
log_std
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
ѓ
log_std/AssignAssignlog_std"log_std/Initializer/random_uniform*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
b
log_std/readIdentitylog_std*
_class
loc:@log_std*
T0*
_output_shapes
:
=
ExpExplog_std/read*
T0*
_output_shapes
:
d
random_normal/shapeConst*
dtype0*
valueB"и     *
_output_shapes
:
Я
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	и
k
random_normal/mulMul"random_normal/RandomStandardNormalExp*
T0*
_output_shapes
:	и
{
random_normalAddrandom_normal/mul(policy_network/fully_connected_1/BiasAdd*
T0*
_output_shapes
:	и
?
Exp_1Explog_std/read*
T0*
_output_shapes
:
§
bMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
©
fMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag_1/batch_shape_tensor/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
†
8MultivariateNormalDiag_1/shapes_from_loc_and_scale/ShapeShape(policy_network/fully_connected_1/BiasAdd*
out_type0*
T0*
_output_shapes
:
Р
FMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ы
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_1Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Т
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ь
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
Њ
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
 *  А?*
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
А
>affine_linear_operator/init/DistributionShape/init/batch_ndimsConst*
dtype0*
value	B : *
_output_shapes
: 
А
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
Б
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
Г
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
valueB:*
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
Г
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
З
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
¬
MultivariateNormalDiag_2/rangeRangeMultivariateNormalDiag_2/sub$MultivariateNormalDiag_2/range/limit$MultivariateNormalDiag_2/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Є
DMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/subSuba_cont(policy_network/fully_connected_1/BiasAdd*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
У
–MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
ю
їMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
Џ
ЧMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
Э
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
џ
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
ф
ЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAddШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xЧMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
ъ
ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPackЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
N*
T0*
_output_shapes
:*

axis 
Ю
ЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePackїMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
N*
T0*
_output_shapes
:*

axis 
∞
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSliceШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
в
ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
Я
ЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1ShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
Ё
ЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/xConst*
dtype0*
value	B : *
_output_shapes
: 
Ь
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1AddЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/xїMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
T0*
_output_shapes
: 
Ь
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2AddШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1>affine_linear_operator/init/DistributionShape/init/batch_ndims*
T0*
_output_shapes
: 
ю
†MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/beginPackШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2*
N*
T0*
_output_shapes
:*

axis 
Ґ
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/sizePack>affine_linear_operator/init/DistributionShape/init/event_ndims*
N*
T0*
_output_shapes
:*

axis 
Є
ЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1SliceЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1†MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/beginЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/size*
Index0*
T0*
_output_shapes
:
 
ЗMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/pick_vector/condConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
Џ
ЖMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
≈
ВMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
П
}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concatConcatV2ЖMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1ВMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ч
~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/ReshapeReshapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
ў
НMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ѕ
ЄMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
и
ЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
А
СMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/ReshapeЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
®
cMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
э
aMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivRealDivcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xExp_1*
T0*
_output_shapes
:
≥
hMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
ф
dMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims
ExpandDimsaMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivhMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
Щ
]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mulMulСMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
„
ФMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
value	B :*
_output_shapes
: 
»
њMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
п
ЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
н
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mulЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ъ
„MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
Е
¬MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
б
ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
щ
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
в
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
Й
ЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAddЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
И
•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPackЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
N*
T0*
_output_shapes
:*

axis 
ђ
§MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePack¬MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
N*
T0*
_output_shapes
:*

axis 
ћ
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSliceЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/begin§MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
й
•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
ы
°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1ShapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
д
°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/xConst*
dtype0*
value	B : *
_output_shapes
: 
±
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1Add°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/x¬MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
T0*
_output_shapes
: 
™
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2AddЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1>affine_linear_operator/init/DistributionShape/init/batch_ndims*
T0*
_output_shapes
: 
М
ІMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/beginPackЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2*
N*
T0*
_output_shapes
:*

axis 
©
¶MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/sizePack>affine_linear_operator/init/DistributionShape/init/event_ndims*
N*
T0*
_output_shapes
:*

axis 
‘
°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1Slice°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1ІMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/begin¶MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/size*
Index0*
T0*
_output_shapes
:
ћ
ЙMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Њ
ДMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concatConcatV2ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shape°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1ЙMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
ь
ЕMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/ReshapeReshapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transposeДMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
©
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsAbsExp_1*
T0*
_output_shapes
:
Ъ
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogLogvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs*
T0*
_output_shapes
:
№
ИMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Њ
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/SumSumvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogИMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
ч
WMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/NegNegvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum*
T0*
_output_shapes
: 
Ч
AMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subSubЕMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape
Normal/loc*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
№
EMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truedivRealDivAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subNormal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ƒ
8MultivariateNormalDiag_3/log_prob/Normal/log_prob/SquareSquareEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/xConst*
dtype0*
valueB
 *   њ*
_output_shapes
: 
к
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mulMul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
k
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/LogLogNormal/scale*
T0*
_output_shapes
: 
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/xConst*
dtype0*
valueB
 *О?k?*
_output_shapes
: 
Ќ
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/addAdd7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/x5MultivariateNormalDiag_3/log_prob/Normal/log_prob/Log*
T0*
_output_shapes
: 
е
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subSub5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul5MultivariateNormalDiag_3/log_prob/Normal/log_prob/add*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
√
%MultivariateNormalDiag_3/log_prob/SumSum5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subMultivariateNormalDiag_2/range*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Џ
%MultivariateNormalDiag_3/log_prob/addAdd%MultivariateNormalDiag_3/log_prob/SumWMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg*
T0*#
_output_shapes
:€€€€€€€€€
_
NegNeg%MultivariateNormalDiag_3/log_prob/add*
T0*#
_output_shapes
:€€€€€€€€€
D
mulMulNegadvtg*
T0*#
_output_shapes
:€€€€€€€€€
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
 *  А?*
_output_shapes
: 
j
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*#
_output_shapes
:€€€€€€€€€
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
і
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
b
gradients/mul_grad/mulMulgradients/Filladvtg*
T0*#
_output_shapes
:€€€€€€€€€
Я
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
b
gradients/mul_grad/mul_1MulNeggradients/Fill*
T0*#
_output_shapes
:€€€€€€€€€
•
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Щ
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
÷
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*#
_output_shapes
:€€€€€€€€€
№
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*#
_output_shapes
:€€€€€€€€€
x
gradients/Neg_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*#
_output_shapes
:€€€€€€€€€
®
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/ShapeShape%MultivariateNormalDiag_3/log_prob/Sum*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€

<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
Ъ
Jgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
г
8gradients/MultivariateNormalDiag_3/log_prob/add_grad/SumSumgradients/Neg_grad/NegJgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
о
<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeReshape8gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape*
_output_shapes
:*
T0*
Tshape0
з
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1Sumgradients/Neg_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
т
>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1Reshape:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Ќ
Egradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_depsNoOp=^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape?^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1
”
Mgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyIdentity<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeF^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*O
_classE
CAloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape*
T0*
_output_shapes
:
„
Ogradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1Identity>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1F^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1*
T0*
_output_shapes
: 
ѓ
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub*
out_type0*
T0*
_output_shapes
:
 
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/SizeConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ч
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/addAddMultivariateNormalDiag_2/range9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ґ
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/modFloorMod8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/add9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Г
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1Shape8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod*
out_type0*
T0*
_output_shapes
:*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape
—
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/startConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B : *
_output_shapes
: 
—
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/deltaConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
щ
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/rangeRange@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/start9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/delta*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*

Tidx0*
_output_shapes
:
–
?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/valueConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
љ
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/FillFill<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/value*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
∆
Bgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitchDynamicStitch:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*
N
ѕ
>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/yConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
»
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/MaximumMaximumBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/y*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Ј
=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordivFloorDiv:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*
_output_shapes
:
Л
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ReshapeReshapeMgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Л
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileTile<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Reshape=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
г
jgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/NegNegOgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
њ
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul*
out_type0*
T0*
_output_shapes
:
П
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
 
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
¶
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumSum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ґ
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
™
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1Sum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Tile\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Њ
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegNegJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
†
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1ReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
э
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1
Ђ
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ч
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1*
T0*
_output_shapes
: 
÷
Лgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
с
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/SizeConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
т
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/addAddИMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
ш
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/modFloorModЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/addКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
ь
Нgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1Const*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
valueB:*
_output_shapes
:
ш
Сgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/startConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B : *
_output_shapes
: 
ш
Сgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/deltaConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ц
Лgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/rangeRangeСgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/startКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/SizeСgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/delta*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*

Tidx0*
_output_shapes
:
ч
Рgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/valueConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
€
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/FillFillНgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1Рgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/value*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
µ
Уgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitchDynamicStitchЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/rangeЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/modЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*
N
ц
Пgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/yConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
У
Нgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/MaximumMaximumУgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitchПgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/y*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Л
Оgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordivFloorDivЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeНgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ћ
Нgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ReshapeReshapejgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/NegУgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
л
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/TileTileНgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ReshapeОgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
Н
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
ƒ
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1Shape8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
out_type0*
T0*
_output_shapes
:
 
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
£
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulMul]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
µ
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumSumHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ь
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
§
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1Mul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ї
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Љ
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1ReshapeJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
э
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1
С
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape*
T0*
_output_shapes
: 
±
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
Рgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal
ReciprocalvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsЛ^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Tile*
T0*
_output_shapes
:
÷
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mulMulКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/TileРgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal*
T0*
_output_shapes
:
ф
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xConst`^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1*
dtype0*
valueB
 *   @*
_output_shapes
: 
£
Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
љ
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Mul_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
њ
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/SignSignExp_1*
T0*
_output_shapes
:
ѕ
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mulMulЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mulКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/Sign*
T0*
_output_shapes
:
џ
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ShapeShapeAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
out_type0*
T0*
_output_shapes
:
Я
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
ъ
jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivRealDivMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Normal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
й
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumSum\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivjgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ж
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ReshapeReshapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
Ё
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNegAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
М
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1RealDivXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNormal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Т
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2RealDiv^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1Normal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
…
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
й
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1SumXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mullgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
“
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1ReshapeZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
≠
egradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_depsNoOp]^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1
л
mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentity\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshapef^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*o
_classe
caloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
„
ogradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependency_1Identity^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1f^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*q
_classg
ecloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
Ь
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeShapeЕMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape*
out_type0*
T0*
_output_shapes
:
Ы
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
о
fgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumSummgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyfgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Џ
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ReshapeReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
ц
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1Summgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyhgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
÷
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegNegVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
ƒ
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1ReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
°
agradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_depsNoOpY^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape[^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1
џ
igradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependencyIdentityXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshapeb^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*k
_classa
_]loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
«
kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1b^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1*
T0*
_output_shapes
: 
ф
Ъgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ShapeShapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
щ
Ьgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshapeigradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependencyЪgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
Ф
єgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutationЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
в
±gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	TransposeЬgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Reshapeєgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ƒ
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ShapeShapeСMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
≈
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
√
Вgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgsBroadcastGradientArgsrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mulMul±gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ѓ
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/SumSumpgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mulВgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ѓ
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ReshapeReshapepgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
ь
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1MulСMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose±gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
і
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1Дgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ґ
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1Reshapergradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
х
}gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_depsNoOpu^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshapew^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1
Ќ
Еgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependencyIdentitytgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*З
_class}
{yloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ѕ
Зgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1Identityvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*Й
_class
}{loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1*
T0*
_output_shapes

:
Ж
≤gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutationЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
љ
™gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	TransposeЕgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency≤gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
√
ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Њ
{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeReshapeЗgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Shape*
_output_shapes
:*
T0*
Tshape0
Ш
Уgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
≠
Хgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshape™gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transposeУgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
є
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
¬
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
ѕ
Жgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shapexgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ђ
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDivRealDiv{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeExp_1*
T0*
_output_shapes
:
Њ
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/SumSumxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDivЖgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
†
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ReshapeReshapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sumvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Б
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegNegcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/x*
T0*
_output_shapes
: 
І
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1RealDivtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegExp_1*
T0*
_output_shapes
:
≠
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2RealDivzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1Exp_1*
T0*
_output_shapes
:
Щ
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mulMul{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Reshapezgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2*
T0*
_output_shapes
:
Њ
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1Sumtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mulИgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
™
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1Reshapevgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
В
Бgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_depsNoOpy^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape{^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1
≈
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependencyIdentityxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ReshapeВ^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*М
_classБ
}loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape*
T0*
_output_shapes
: 
–
Лgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1Identityzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1В^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*П
_classД
Бloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1*
T0*
_output_shapes
:
А
gradients/AddNAddNЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mulЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1*Я
_classФ
СОloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mul*
T0*
_output_shapes
:*
N
[
gradients/Exp_1_grad/mulMulgradients/AddNExp_1*
T0*
_output_shapes
:
Я
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ShapeShapea_cont*
out_type0*
T0*
_output_shapes
:
√
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1Shape(policy_network/fully_connected_1/BiasAdd*
out_type0*
T0*
_output_shapes
:
ч
igradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgsBroadcastGradientArgsYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
°
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumSumХgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapeigradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
г
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ReshapeReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
•
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1SumХgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapekgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
№
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/NegNegYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1*
T0*
_output_shapes
:
ё
]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1ReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Neg[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
™
dgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_depsNoOp\^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape^^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1
з
lgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependencyIdentity[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshapee^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*n
_classd
b`loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
д
ngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1Identity]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1e^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
ю
Cgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGradngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*
data_formatNHWC
З
Hgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOpo^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1D^gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad
ї
Pgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentityngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1I^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
л
Rgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
†
=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
П
?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1MatMul#policy_network/fully_connected/ReluPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
—
Ggradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1
и
Ogradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
е
Qgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
п
;gradients/policy_network/fully_connected/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency#policy_network/fully_connected/Relu*
T0*'
_output_shapes
:€€€€€€€€€
…
Agradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
–
Fgradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/Relu_grad/ReluGradB^gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad
в
Ngradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/Relu_grad/ReluGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:€€€€€€€€€
г
Pgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
Ъ
;gradients/policy_network/fully_connected/MatMul_grad/MatMulMatMulNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency+policy_network/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
й
=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1MatMuloNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
Ћ
Egradients/policy_network/fully_connected/MatMul_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/MatMul_grad/MatMul>^gradients/policy_network/fully_connected/MatMul_grad/MatMul_1
а
Mgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/MatMul_grad/MatMulF^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
Ё
Ogradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1Identity=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1F^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
z
beta1_power/initial_valueConst*
dtype0*
_class
loc:@log_std*
valueB
 *fff?*
_output_shapes
: 
Л
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
™
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
 *wЊ?*
_output_shapes
: 
Л
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
™
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
Ќ
=policy_network/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
Џ
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
µ
2policy_network/fully_connected/weights/Adam/AssignAssign+policy_network/fully_connected/weights/Adam=policy_network/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
Ќ
0policy_network/fully_connected/weights/Adam/readIdentity+policy_network/fully_connected/weights/Adam*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
ѕ
?policy_network/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
№
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
ї
4policy_network/fully_connected/weights/Adam_1/AssignAssign-policy_network/fully_connected/weights/Adam_1?policy_network/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
—
2policy_network/fully_connected/weights/Adam_1/readIdentity-policy_network/fully_connected/weights/Adam_1*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
√
<policy_network/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
–
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
≠
1policy_network/fully_connected/biases/Adam/AssignAssign*policy_network/fully_connected/biases/Adam<policy_network/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
∆
/policy_network/fully_connected/biases/Adam/readIdentity*policy_network/fully_connected/biases/Adam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
≈
>policy_network/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
“
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
≥
3policy_network/fully_connected/biases/Adam_1/AssignAssign,policy_network/fully_connected/biases/Adam_1>policy_network/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
 
1policy_network/fully_connected/biases/Adam_1/readIdentity,policy_network/fully_connected/biases/Adam_1*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
—
?policy_network/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
ё
-policy_network/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
љ
4policy_network/fully_connected_1/weights/Adam/AssignAssign-policy_network/fully_connected_1/weights/Adam?policy_network/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
”
2policy_network/fully_connected_1/weights/Adam/readIdentity-policy_network/fully_connected_1/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
”
Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
а
/policy_network/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
√
6policy_network/fully_connected_1/weights/Adam_1/AssignAssign/policy_network/fully_connected_1/weights/Adam_1Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
„
4policy_network/fully_connected_1/weights/Adam_1/readIdentity/policy_network/fully_connected_1/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
«
>policy_network/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
‘
,policy_network/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
µ
3policy_network/fully_connected_1/biases/Adam/AssignAssign,policy_network/fully_connected_1/biases/Adam>policy_network/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
ћ
1policy_network/fully_connected_1/biases/Adam/readIdentity,policy_network/fully_connected_1/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
…
@policy_network/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
÷
.policy_network/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
ї
5policy_network/fully_connected_1/biases/Adam_1/AssignAssign.policy_network/fully_connected_1/biases/Adam_1@policy_network/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
–
3policy_network/fully_connected_1/biases/Adam_1/readIdentity.policy_network/fully_connected_1/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
З
log_std/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
Ф
log_std/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
µ
log_std/Adam/AssignAssignlog_std/Adamlog_std/Adam/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
l
log_std/Adam/readIdentitylog_std/Adam*
_class
loc:@log_std*
T0*
_output_shapes
:
Й
 log_std/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
Ц
log_std/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
ї
log_std/Adam_1/AssignAssignlog_std/Adam_1 log_std/Adam_1/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
p
log_std/Adam_1/readIdentitylog_std/Adam_1*
_class
loc:@log_std*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *П¬х<*
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
 *wЊ?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
З
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
€
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
У
>Adam/update_policy_network/fully_connected_1/weights/ApplyAdam	ApplyAdam(policy_network/fully_connected_1/weights-policy_network/fully_connected_1/weights/Adam/policy_network/fully_connected_1/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:
Л
=Adam/update_policy_network/fully_connected_1/biases/ApplyAdam	ApplyAdam'policy_network/fully_connected_1/biases,policy_network/fully_connected_1/biases/Adam.policy_network/fully_connected_1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
:
±
Adam/update_log_std/ApplyAdam	ApplyAdamlog_stdlog_std/Adamlog_std/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/Exp_1_grad/mul*
use_nesterov( *
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
:
И
Adam/mulMulbeta1_power/read
Adam/beta1=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
Т
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
К

Adam/mul_1Mulbeta2_power/read
Adam/beta2=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
Ц
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
»
AdamNoOp=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/update_log_std/ApplyAdam^Adam/Assign^Adam/Assign_1
«
Abaseline/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB"      *
_output_shapes
:
є
?baseline/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *М7њ*
_output_shapes
: 
є
?baseline/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *М7?*
_output_shapes
: 
°
Ibaseline/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformAbaseline/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@baseline/fully_connected/weights
Ю
?baseline/fully_connected/weights/Initializer/random_uniform/subSub?baseline/fully_connected/weights/Initializer/random_uniform/max?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes
: 
∞
?baseline/fully_connected/weights/Initializer/random_uniform/mulMulIbaseline/fully_connected/weights/Initializer/random_uniform/RandomUniform?baseline/fully_connected/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
Ґ
;baseline/fully_connected/weights/Initializer/random_uniformAdd?baseline/fully_connected/weights/Initializer/random_uniform/mul?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
…
 baseline/fully_connected/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
Ч
'baseline/fully_connected/weights/AssignAssign baseline/fully_connected/weights;baseline/fully_connected/weights/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
±
%baseline/fully_connected/weights/readIdentity baseline/fully_connected/weights*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
≤
1baseline/fully_connected/biases/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB*    *
_output_shapes
:
њ
baseline/fully_connected/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
Ж
&baseline/fully_connected/biases/AssignAssignbaseline/fully_connected/biases1baseline/fully_connected/biases/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
™
$baseline/fully_connected/biases/readIdentitybaseline/fully_connected/biases*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
:
Ђ
baseline/fully_connected/MatMulMatMulo%baseline/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
ї
 baseline/fully_connected/BiasAddBiasAddbaseline/fully_connected/MatMul$baseline/fully_connected/biases/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
y
baseline/fully_connected/ReluRelu baseline/fully_connected/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Ћ
Cbaseline/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB"      *
_output_shapes
:
љ
Abaseline/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *0њ*
_output_shapes
: 
љ
Abaseline/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *0?*
_output_shapes
: 
І
Kbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformCbaseline/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*5
_class+
)'loc:@baseline/fully_connected_1/weights
¶
Abaseline/fully_connected_1/weights/Initializer/random_uniform/subSubAbaseline/fully_connected_1/weights/Initializer/random_uniform/maxAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes
: 
Є
Abaseline/fully_connected_1/weights/Initializer/random_uniform/mulMulKbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformAbaseline/fully_connected_1/weights/Initializer/random_uniform/sub*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
™
=baseline/fully_connected_1/weights/Initializer/random_uniformAddAbaseline/fully_connected_1/weights/Initializer/random_uniform/mulAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
Ќ
"baseline/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
Я
)baseline/fully_connected_1/weights/AssignAssign"baseline/fully_connected_1/weights=baseline/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
Ј
'baseline/fully_connected_1/weights/readIdentity"baseline/fully_connected_1/weights*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
ґ
3baseline/fully_connected_1/biases/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB*    *
_output_shapes
:
√
!baseline/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
О
(baseline/fully_connected_1/biases/AssignAssign!baseline/fully_connected_1/biases3baseline/fully_connected_1/biases/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
∞
&baseline/fully_connected_1/biases/readIdentity!baseline/fully_connected_1/biases*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
:
Ћ
!baseline/fully_connected_1/MatMulMatMulbaseline/fully_connected/Relu'baseline/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
Ѕ
"baseline/fully_connected_1/BiasAddBiasAdd!baseline/fully_connected_1/MatMul&baseline/fully_connected_1/biases/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
e

baseline_1Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
m
SqueezeSqueeze"baseline/fully_connected_1/BiasAdd*
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
 *  А?*
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
°
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
В
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
∆
1mean_squared_error/assert_broadcastable/is_scalarEqual3mean_squared_error/assert_broadcastable/is_scalar/x4mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
–
=mean_squared_error/assert_broadcastable/is_valid_shape/SwitchSwitch1mean_squared_error/assert_broadcastable/is_scalar1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
≠
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_tIdentity?mean_squared_error/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
Ђ
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_fIdentity=mean_squared_error/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
Ю
>mean_squared_error/assert_broadcastable/is_valid_shape/pred_idIdentity1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
•
?mean_squared_error/assert_broadcastable/is_valid_shape/Switch_1Switch1mean_squared_error/assert_broadcastable/is_scalar>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
’
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualdmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchfmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
ќ
dmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch3mean_squared_error/assert_broadcastable/values/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*F
_class<
:8loc:@mean_squared_error/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
“
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch4mean_squared_error/assert_broadcastable/weights/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
¬
Wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
б
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
я
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityWmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
д
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Ч
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
£
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimswmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
щ
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch4mean_squared_error/assert_broadcastable/values/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
‘
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchsmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapelmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
П
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
Л
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:€€€€€€€€€
Л
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
Е
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimskmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*

Tidx0*'
_output_shapes
:€€€€€€€€€*
T0
Щ
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ю
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
з
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch5mean_squared_error/assert_broadcastable/weights/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
√
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchumean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
з
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationnmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:*
T0*
set_operationa-b
©
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
Б
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
д
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualcmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xrmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
±
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*p
_classf
dbloc:@mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
…
Vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N
М
<mean_squared_error/assert_broadcastable/is_valid_shape/MergeMergeVmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeAmean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
Х
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
°
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
Ц
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
г
:mean_squared_error/assert_broadcastable/AssertGuard/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
І
<mean_squared_error/assert_broadcastable/AssertGuard/switch_tIdentity<mean_squared_error/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
•
<mean_squared_error/assert_broadcastable/AssertGuard/switch_fIdentity:mean_squared_error/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
¶
;mean_squared_error/assert_broadcastable/AssertGuard/pred_idIdentity<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 

8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp=^mean_squared_error/assert_broadcastable/AssertGuard/switch_t
љ
Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependencyIdentity<mean_squared_error/assert_broadcastable/AssertGuard/switch_t9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
и
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
ѕ
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
т
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
_output_shapes
: 
ќ
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
з
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
Ћ
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
Н
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertAmean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchAmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

Ї
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
≤
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Switch5mean_squared_error/assert_broadcastable/weights/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
∆
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Switch4mean_squared_error/assert_broadcastable/values/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
¶
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3Switch1mean_squared_error/assert_broadcastable/is_scalar;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
Ѕ
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Identity<mean_squared_error/assert_broadcastable/AssertGuard/switch_f;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
А
9mean_squared_error/assert_broadcastable/AssertGuard/MergeMergeHmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
Я
mean_squared_error/ToFloat_3/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Ж
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
X
mean_squared_error/RankRankmean_squared_error/Mul*
T0*
_output_shapes
: 
Ь
mean_squared_error/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
Ь
mean_squared_error/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
Ђ
mean_squared_error/rangeRangemean_squared_error/range/startmean_squared_error/Rankmean_squared_error/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Н
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/range*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
І
&mean_squared_error/num_present/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
Ц
$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat_3/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
™
)mean_squared_error/num_present/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
≠
.mean_squared_error/num_present/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
ѓ
.mean_squared_error/num_present/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
±
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*
_output_shapes
: 
Ћ
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
“
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
–
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
ы
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
№
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
: 
ѕ
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
†
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
™
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
й
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
з
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
Џ
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Э
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
±
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualВmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchДmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
«
Вmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*d
_classZ
XVloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
Ћ
Дmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
Ь
umean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Э
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Ы
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityumean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
†
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Р
Оmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
А
Кmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsХmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Оmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
т
Сmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchСmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
џ
Пmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapeКmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
И
Пmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
и
Йmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillПmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeПmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:€€€€€€€€€
Д
Лmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
Б
Жmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Кmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsЙmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeЛmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*

Tidx0*'
_output_shapes
:€€€€€€€€€*
T0
Т
Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
ы
Мmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЧmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
а
Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
љ
Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchУmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
ƒ
Шmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationМmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Жmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:*
T0*
set_operationa-b
з
Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeЪmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
ъ
Бmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
ј
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualБmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xРmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
≠
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*С
_classЖ
ГАloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
£
tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergewmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N
ж
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergetmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
п
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
Ў
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
с
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
„
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
р
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
‘
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
љ
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
г
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
б
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
в
Ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
ч
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
µ
dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
а
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
«
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
а
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
∆
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
я
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
√
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
„
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

≤
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
™
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
Њ
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ю
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
є
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
Џ
Wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergefmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
√
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
Ы
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
й
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*
_output_shapes
:
љ
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
~
#mean_squared_error/num_present/RankRank0mean_squared_error/num_present/broadcast_weights*
T0*
_output_shapes
: 
®
*mean_squared_error/num_present/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
®
*mean_squared_error/num_present/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
џ
$mean_squared_error/num_present/rangeRange*mean_squared_error/num_present/range/start#mean_squared_error/num_present/Rank*mean_squared_error/num_present/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
ї
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/range*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ч
mean_squared_error/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
П
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Э
mean_squared_error/Greater/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
Д
mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
T0*
_output_shapes
: 
Ы
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
°
"mean_squared_error/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
£
"mean_squared_error/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Н
mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*
_output_shapes
: 
Ь
mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ю
mean_squared_error/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
Ц
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
 *  А?*
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
њ
0gradients_1/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients_1/Fill4gradients_1/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
Ѕ
2gradients_1/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater4gradients_1/mean_squared_error/value_grad/zeros_likegradients_1/Fill*
T0*
_output_shapes
: 
™
:gradients_1/mean_squared_error/value_grad/tuple/group_depsNoOp1^gradients_1/mean_squared_error/value_grad/Select3^gradients_1/mean_squared_error/value_grad/Select_1
£
Bgradients_1/mean_squared_error/value_grad/tuple/control_dependencyIdentity0gradients_1/mean_squared_error/value_grad/Select;^gradients_1/mean_squared_error/value_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/mean_squared_error/value_grad/Select*
T0*
_output_shapes
: 
©
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
у
=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/div_grad/Shape/gradients_1/mean_squared_error/div_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ї
/gradients_1/mean_squared_error/div_grad/RealDivRealDivBgradients_1/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
в
+gradients_1/mean_squared_error/div_grad/SumSum/gradients_1/mean_squared_error/div_grad/RealDiv=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
≈
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
•
1gradients_1/mean_squared_error/div_grad/RealDiv_1RealDiv+gradients_1/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
Ђ
1gradients_1/mean_squared_error/div_grad/RealDiv_2RealDiv1gradients_1/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
 
+gradients_1/mean_squared_error/div_grad/mulMulBgradients_1/mean_squared_error/value_grad/tuple/control_dependency1gradients_1/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
в
-gradients_1/mean_squared_error/div_grad/Sum_1Sum+gradients_1/mean_squared_error/div_grad/mul?gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ћ
1gradients_1/mean_squared_error/div_grad/Reshape_1Reshape-gradients_1/mean_squared_error/div_grad/Sum_1/gradients_1/mean_squared_error/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
¶
8gradients_1/mean_squared_error/div_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/div_grad/Reshape2^gradients_1/mean_squared_error/div_grad/Reshape_1
Э
@gradients_1/mean_squared_error/div_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/div_grad/Reshape9^gradients_1/mean_squared_error/div_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/div_grad/Reshape*
T0*
_output_shapes
: 
£
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
ж
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
÷
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
с
1gradients_1/mean_squared_error/Select_grad/SelectSelectmean_squared_error/EqualBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_15gradients_1/mean_squared_error/Select_grad/zeros_like*
T0*
_output_shapes
: 
у
3gradients_1/mean_squared_error/Select_grad/Select_1Selectmean_squared_error/Equal5gradients_1/mean_squared_error/Select_grad/zeros_likeBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
≠
;gradients_1/mean_squared_error/Select_grad/tuple/group_depsNoOp2^gradients_1/mean_squared_error/Select_grad/Select4^gradients_1/mean_squared_error/Select_grad/Select_1
І
Cgradients_1/mean_squared_error/Select_grad/tuple/control_dependencyIdentity1gradients_1/mean_squared_error/Select_grad/Select<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Select_grad/Select*
T0*
_output_shapes
: 
≠
Egradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1Identity3gradients_1/mean_squared_error/Select_grad/Select_1<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*F
_class<
:8loc:@gradients_1/mean_squared_error/Select_grad/Select_1*
T0*
_output_shapes
: 
М
-gradients_1/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
÷
,gradients_1/mean_squared_error/Sum_grad/SizeSize-gradients_1/mean_squared_error/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
: *@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape
к
+gradients_1/mean_squared_error/Sum_grad/addAddmean_squared_error/range,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
В
+gradients_1/mean_squared_error/Sum_grad/modFloorMod+gradients_1/mean_squared_error/Sum_grad/add,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
№
/gradients_1/mean_squared_error/Sum_grad/Shape_1Shape+gradients_1/mean_squared_error/Sum_grad/mod*
out_type0*
T0*
_output_shapes
:*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape
Ј
3gradients_1/mean_squared_error/Sum_grad/range/startConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B : *
_output_shapes
: 
Ј
3gradients_1/mean_squared_error/Sum_grad/range/deltaConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ѕ
-gradients_1/mean_squared_error/Sum_grad/rangeRange3gradients_1/mean_squared_error/Sum_grad/range/start,gradients_1/mean_squared_error/Sum_grad/Size3gradients_1/mean_squared_error/Sum_grad/range/delta*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*

Tidx0*#
_output_shapes
:€€€€€€€€€
ґ
2gradients_1/mean_squared_error/Sum_grad/Fill/valueConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Й
,gradients_1/mean_squared_error/Sum_grad/FillFill/gradients_1/mean_squared_error/Sum_grad/Shape_12gradients_1/mean_squared_error/Sum_grad/Fill/value*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ш
5gradients_1/mean_squared_error/Sum_grad/DynamicStitchDynamicStitch-gradients_1/mean_squared_error/Sum_grad/range+gradients_1/mean_squared_error/Sum_grad/mod-gradients_1/mean_squared_error/Sum_grad/Shape,gradients_1/mean_squared_error/Sum_grad/Fill*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*
N
µ
1gradients_1/mean_squared_error/Sum_grad/Maximum/yConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ф
/gradients_1/mean_squared_error/Sum_grad/MaximumMaximum5gradients_1/mean_squared_error/Sum_grad/DynamicStitch1gradients_1/mean_squared_error/Sum_grad/Maximum/y*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
М
0gradients_1/mean_squared_error/Sum_grad/floordivFloorDiv-gradients_1/mean_squared_error/Sum_grad/Shape/gradients_1/mean_squared_error/Sum_grad/Maximum*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
“
/gradients_1/mean_squared_error/Sum_grad/ReshapeReshape.gradients_1/mean_squared_error/Sum_1_grad/Tile5gradients_1/mean_squared_error/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
ћ
,gradients_1/mean_squared_error/Sum_grad/TileTile/gradients_1/mean_squared_error/Sum_grad/Reshape0gradients_1/mean_squared_error/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
Ѓ
5gradients_1/mean_squared_error/num_present_grad/ShapeShape0mean_squared_error/num_present/broadcast_weights*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
о
4gradients_1/mean_squared_error/num_present_grad/SizeSize5gradients_1/mean_squared_error/num_present_grad/Shape*
out_type0*
T0*
_output_shapes
: *H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape
О
3gradients_1/mean_squared_error/num_present_grad/addAdd$mean_squared_error/num_present/range4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Ґ
3gradients_1/mean_squared_error/num_present_grad/modFloorMod3gradients_1/mean_squared_error/num_present_grad/add4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ф
7gradients_1/mean_squared_error/num_present_grad/Shape_1Shape3gradients_1/mean_squared_error/num_present_grad/mod*
out_type0*
T0*
_output_shapes
:*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape
«
;gradients_1/mean_squared_error/num_present_grad/range/startConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B : *
_output_shapes
: 
«
;gradients_1/mean_squared_error/num_present_grad/range/deltaConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
й
5gradients_1/mean_squared_error/num_present_grad/rangeRange;gradients_1/mean_squared_error/num_present_grad/range/start4gradients_1/mean_squared_error/num_present_grad/Size;gradients_1/mean_squared_error/num_present_grad/range/delta*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*

Tidx0*#
_output_shapes
:€€€€€€€€€
∆
:gradients_1/mean_squared_error/num_present_grad/Fill/valueConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
©
4gradients_1/mean_squared_error/num_present_grad/FillFill7gradients_1/mean_squared_error/num_present_grad/Shape_1:gradients_1/mean_squared_error/num_present_grad/Fill/value*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
®
=gradients_1/mean_squared_error/num_present_grad/DynamicStitchDynamicStitch5gradients_1/mean_squared_error/num_present_grad/range3gradients_1/mean_squared_error/num_present_grad/mod5gradients_1/mean_squared_error/num_present_grad/Shape4gradients_1/mean_squared_error/num_present_grad/Fill*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*
N
≈
9gradients_1/mean_squared_error/num_present_grad/Maximum/yConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
і
7gradients_1/mean_squared_error/num_present_grad/MaximumMaximum=gradients_1/mean_squared_error/num_present_grad/DynamicStitch9gradients_1/mean_squared_error/num_present_grad/Maximum/y*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ђ
8gradients_1/mean_squared_error/num_present_grad/floordivFloorDiv5gradients_1/mean_squared_error/num_present_grad/Shape7gradients_1/mean_squared_error/num_present_grad/Maximum*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
щ
7gradients_1/mean_squared_error/num_present_grad/ReshapeReshapeEgradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1=gradients_1/mean_squared_error/num_present_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
д
4gradients_1/mean_squared_error/num_present_grad/TileTile7gradients_1/mean_squared_error/num_present_grad/Reshape8gradients_1/mean_squared_error/num_present_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
Ъ
-gradients_1/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
r
/gradients_1/mean_squared_error/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
у
=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/Mul_grad/Shape/gradients_1/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
£
+gradients_1/mean_squared_error/Mul_grad/mulMul,gradients_1/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
ё
+gradients_1/mean_squared_error/Mul_grad/SumSum+gradients_1/mean_squared_error/Mul_grad/mul=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
«
/gradients_1/mean_squared_error/Mul_grad/ReshapeReshape+gradients_1/mean_squared_error/Mul_grad/Sum-gradients_1/mean_squared_error/Mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
Ђ
-gradients_1/mean_squared_error/Mul_grad/mul_1Mul$mean_squared_error/SquaredDifference,gradients_1/mean_squared_error/Sum_grad/Tile*
T0*
_output_shapes
:
д
-gradients_1/mean_squared_error/Mul_grad/Sum_1Sum-gradients_1/mean_squared_error/Mul_grad/mul_1?gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ћ
1gradients_1/mean_squared_error/Mul_grad/Reshape_1Reshape-gradients_1/mean_squared_error/Mul_grad/Sum_1/gradients_1/mean_squared_error/Mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
¶
8gradients_1/mean_squared_error/Mul_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/Mul_grad/Reshape2^gradients_1/mean_squared_error/Mul_grad/Reshape_1
Я
@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/Mul_grad/Reshape9^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/Mul_grad/Reshape*
T0*
_output_shapes
:
£
Bgradients_1/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity1gradients_1/mean_squared_error/Mul_grad/Reshape_19^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Mul_grad/Reshape_1*
T0*
_output_shapes
: 
К
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
ћ
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1Shape:mean_squared_error/num_present/broadcast_weights/ones_like*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
Ѕ
Wgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
б
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulMul4gradients_1/mean_squared_error/num_present_grad/Tile:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
ђ
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumSumEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulWgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeReshapeEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ќ
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Mul%mean_squared_error/num_present/Select4gradients_1/mean_squared_error/num_present_grad/Tile*
T0*
_output_shapes
:
≤
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Ygradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ы
Kgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1ReshapeGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
ф
Rgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_depsNoOpJ^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeL^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1
Е
Zgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeS^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape*
T0*
_output_shapes
: 
Н
\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityKgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1S^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1*
T0*
_output_shapes
:
„
Pgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankRank\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Щ
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Щ
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
П
Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/rangeRangeWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startPgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
≈
Ogradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/SumSum\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Е
;gradients_1/mean_squared_error/SquaredDifference_grad/ShapeShape
baseline_1*
out_type0*
T0*
_output_shapes
:
Н
=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1ShapeSqueeze*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
Э
Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/mean_squared_error/SquaredDifference_grad/Shape=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ƒ
<gradients_1/mean_squared_error/SquaredDifference_grad/scalarConstA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
dtype0*
valueB
 *   @*
_output_shapes
: 
г
9gradients_1/mean_squared_error/SquaredDifference_grad/mulMul<gradients_1/mean_squared_error/SquaredDifference_grad/scalar@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
ї
9gradients_1/mean_squared_error/SquaredDifference_grad/subSub
baseline_1SqueezeA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
џ
;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mul9gradients_1/mean_squared_error/SquaredDifference_grad/mul9gradients_1/mean_squared_error/SquaredDifference_grad/sub*
T0*
_output_shapes
:
К
9gradients_1/mean_squared_error/SquaredDifference_grad/SumSum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ь
=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeReshape9gradients_1/mean_squared_error/SquaredDifference_grad/Sum;gradients_1/mean_squared_error/SquaredDifference_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
О
;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1Sum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ч
?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
§
9gradients_1/mean_squared_error/SquaredDifference_grad/NegNeg?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes
:
 
Fgradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp>^gradients_1/mean_squared_error/SquaredDifference_grad/Reshape:^gradients_1/mean_squared_error/SquaredDifference_grad/Neg
в
Ngradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/mean_squared_error/SquaredDifference_grad/Reshape*
T0*#
_output_shapes
:€€€€€€€€€
—
Pgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity9gradients_1/mean_squared_error/SquaredDifference_grad/NegG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/mean_squared_error/SquaredDifference_grad/Neg*
T0*
_output_shapes
:
А
gradients_1/Squeeze_grad/ShapeShape"baseline/fully_connected_1/BiasAdd*
out_type0*
T0*
_output_shapes
:
Ё
 gradients_1/Squeeze_grad/ReshapeReshapePgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1gradients_1/Squeeze_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
ђ
?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad gradients_1/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
±
Dgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp!^gradients_1/Squeeze_grad/Reshape@^gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad
®
Lgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity gradients_1/Squeeze_grad/ReshapeE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Squeeze_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€
џ
Ngradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
Т
9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulMatMulLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency'baseline/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
Б
;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1MatMulbaseline/fully_connected/ReluLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
≈
Cgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul<^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1
Ў
Kgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulD^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
’
Mgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1D^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
б
7gradients_1/baseline/fully_connected/Relu_grad/ReluGradReluGradKgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencybaseline/fully_connected/Relu*
T0*'
_output_shapes
:€€€€€€€€€
Ѕ
=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
ƒ
Bgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/Relu_grad/ReluGrad>^gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad
“
Jgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/Relu_grad/ReluGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:€€€€€€€€€
”
Lgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
М
7gradients_1/baseline/fully_connected/MatMul_grad/MatMulMatMulJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency%baseline/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
б
9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1MatMuloJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
њ
Agradients_1/baseline/fully_connected/MatMul_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/MatMul_grad/MatMul:^gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1
–
Igradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/MatMul_grad/MatMulB^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
Ќ
Kgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1Identity9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1B^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
Ф
beta1_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *fff?*
_output_shapes
: 
•
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
»
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
В
beta1_power_1/readIdentitybeta1_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
Ф
beta2_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *wЊ?*
_output_shapes
: 
•
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
»
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
В
beta2_power_1/readIdentitybeta2_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
Ѕ
7baseline/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB*    *
_output_shapes

:
ќ
%baseline/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
Э
,baseline/fully_connected/weights/Adam/AssignAssign%baseline/fully_connected/weights/Adam7baseline/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
ї
*baseline/fully_connected/weights/Adam/readIdentity%baseline/fully_connected/weights/Adam*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
√
9baseline/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB*    *
_output_shapes

:
–
'baseline/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
£
.baseline/fully_connected/weights/Adam_1/AssignAssign'baseline/fully_connected/weights/Adam_19baseline/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
њ
,baseline/fully_connected/weights/Adam_1/readIdentity'baseline/fully_connected/weights/Adam_1*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
Ј
6baseline/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB*    *
_output_shapes
:
ƒ
$baseline/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
Х
+baseline/fully_connected/biases/Adam/AssignAssign$baseline/fully_connected/biases/Adam6baseline/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
і
)baseline/fully_connected/biases/Adam/readIdentity$baseline/fully_connected/biases/Adam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
:
є
8baseline/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB*    *
_output_shapes
:
∆
&baseline/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
Ы
-baseline/fully_connected/biases/Adam_1/AssignAssign&baseline/fully_connected/biases/Adam_18baseline/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
Є
+baseline/fully_connected/biases/Adam_1/readIdentity&baseline/fully_connected/biases/Adam_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
:
≈
9baseline/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB*    *
_output_shapes

:
“
'baseline/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
•
.baseline/fully_connected_1/weights/Adam/AssignAssign'baseline/fully_connected_1/weights/Adam9baseline/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
Ѕ
,baseline/fully_connected_1/weights/Adam/readIdentity'baseline/fully_connected_1/weights/Adam*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
«
;baseline/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB*    *
_output_shapes

:
‘
)baseline/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
Ђ
0baseline/fully_connected_1/weights/Adam_1/AssignAssign)baseline/fully_connected_1/weights/Adam_1;baseline/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
≈
.baseline/fully_connected_1/weights/Adam_1/readIdentity)baseline/fully_connected_1/weights/Adam_1*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
ї
8baseline/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB*    *
_output_shapes
:
»
&baseline/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
Э
-baseline/fully_connected_1/biases/Adam/AssignAssign&baseline/fully_connected_1/biases/Adam8baseline/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
Ї
+baseline/fully_connected_1/biases/Adam/readIdentity&baseline/fully_connected_1/biases/Adam*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
:
љ
:baseline/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB*    *
_output_shapes
:
 
(baseline/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
£
/baseline/fully_connected_1/biases/Adam_1/AssignAssign(baseline/fully_connected_1/biases/Adam_1:baseline/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
Њ
-baseline/fully_connected_1/biases/Adam_1/readIdentity(baseline/fully_connected_1/biases/Adam_1*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
:
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *П¬х<*
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
 *wЊ?*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
у
8Adam_1/update_baseline/fully_connected/weights/ApplyAdam	ApplyAdam baseline/fully_connected/weights%baseline/fully_connected/weights/Adam'baseline/fully_connected/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonKgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking( *
T0*
_output_shapes

:
л
7Adam_1/update_baseline/fully_connected/biases/ApplyAdam	ApplyAdambaseline/fully_connected/biases$baseline/fully_connected/biases/Adam&baseline/fully_connected/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonLgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
:
€
:Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam	ApplyAdam"baseline/fully_connected_1/weights'baseline/fully_connected_1/weights/Adam)baseline/fully_connected_1/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonMgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:
ч
9Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam	ApplyAdam!baseline/fully_connected_1/biases&baseline/fully_connected_1/biases/Adam(baseline/fully_connected_1/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
:
ц

Adam_1/mulMulbeta1_power_1/readAdam_1/beta19^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
∞
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
ш
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta29^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
і
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
Ю
Adam_1NoOp9^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam^Adam_1/Assign^Adam_1/Assign_1
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
N"д≠З	{‘     ÷фМx	#-Ќг@§÷AJо®
в&¬&
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
2	АР
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
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
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
є
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
Р
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

2	Р
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
2	Р
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
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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

2	Р
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ц
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
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.5.02v1.5.0-0-g37aa430d84рЙ
d
oPlaceholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
{
a_contPlaceholder*
dtype0*%
shape:€€€€€€€€€€€€€€€€€€*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
`
advtgPlaceholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
”
Gpolicy_network/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB"      *
_output_shapes
:
≈
Epolicy_network/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *М7њ*
_output_shapes
: 
≈
Epolicy_network/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB
 *М7?*
_output_shapes
: 
≥
Opolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformGpolicy_network/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*9
_class/
-+loc:@policy_network/fully_connected/weights
ґ
Epolicy_network/fully_connected/weights/Initializer/random_uniform/subSubEpolicy_network/fully_connected/weights/Initializer/random_uniform/maxEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes
: 
»
Epolicy_network/fully_connected/weights/Initializer/random_uniform/mulMulOpolicy_network/fully_connected/weights/Initializer/random_uniform/RandomUniformEpolicy_network/fully_connected/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
Ї
Apolicy_network/fully_connected/weights/Initializer/random_uniformAddEpolicy_network/fully_connected/weights/Initializer/random_uniform/mulEpolicy_network/fully_connected/weights/Initializer/random_uniform/min*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
’
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
ѓ
-policy_network/fully_connected/weights/AssignAssign&policy_network/fully_connected/weightsApolicy_network/fully_connected/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
√
+policy_network/fully_connected/weights/readIdentity&policy_network/fully_connected/weights*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
Њ
7policy_network/fully_connected/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
Ћ
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
Ю
,policy_network/fully_connected/biases/AssignAssign%policy_network/fully_connected/biases7policy_network/fully_connected/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
Љ
*policy_network/fully_connected/biases/readIdentity%policy_network/fully_connected/biases*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
Ј
%policy_network/fully_connected/MatMulMatMulo+policy_network/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
Ќ
&policy_network/fully_connected/BiasAddBiasAdd%policy_network/fully_connected/MatMul*policy_network/fully_connected/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€
Е
#policy_network/fully_connected/ReluRelu&policy_network/fully_connected/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
„
Ipolicy_network/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB"      *
_output_shapes
:
…
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *0њ*
_output_shapes
: 
…
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB
 *0?*
_output_shapes
: 
є
Qpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformIpolicy_network/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*;
_class1
/-loc:@policy_network/fully_connected_1/weights
Њ
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/subSubGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/maxGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes
: 
–
Gpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulMulQpolicy_network/fully_connected_1/weights/Initializer/random_uniform/RandomUniformGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
¬
Cpolicy_network/fully_connected_1/weights/Initializer/random_uniformAddGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/mulGpolicy_network/fully_connected_1/weights/Initializer/random_uniform/min*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
ў
(policy_network/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
Ј
/policy_network/fully_connected_1/weights/AssignAssign(policy_network/fully_connected_1/weightsCpolicy_network/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
…
-policy_network/fully_connected_1/weights/readIdentity(policy_network/fully_connected_1/weights*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
¬
9policy_network/fully_connected_1/biases/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
ѕ
'policy_network/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
¶
.policy_network/fully_connected_1/biases/AssignAssign'policy_network/fully_connected_1/biases9policy_network/fully_connected_1/biases/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
¬
,policy_network/fully_connected_1/biases/readIdentity'policy_network/fully_connected_1/biases*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
Ё
'policy_network/fully_connected_1/MatMulMatMul#policy_network/fully_connected/Relu-policy_network/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
”
(policy_network/fully_connected_1/BiasAddBiasAdd'policy_network/fully_connected_1/MatMul,policy_network/fully_connected_1/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€
О
(log_std/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@log_std*
valueB:*
_output_shapes
:
З
&log_std/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@log_std*
valueB
 *„≥Ёњ*
_output_shapes
: 
З
&log_std/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@log_std*
valueB
 *„≥Ё?*
_output_shapes
: 
“
0log_std/Initializer/random_uniform/RandomUniformRandomUniform(log_std/Initializer/random_uniform/shape*
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*
_class
loc:@log_std
Ї
&log_std/Initializer/random_uniform/subSub&log_std/Initializer/random_uniform/max&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
: 
»
&log_std/Initializer/random_uniform/mulMul0log_std/Initializer/random_uniform/RandomUniform&log_std/Initializer/random_uniform/sub*
_class
loc:@log_std*
T0*
_output_shapes
:
Ї
"log_std/Initializer/random_uniformAdd&log_std/Initializer/random_uniform/mul&log_std/Initializer/random_uniform/min*
_class
loc:@log_std*
T0*
_output_shapes
:
П
log_std
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
ѓ
log_std/AssignAssignlog_std"log_std/Initializer/random_uniform*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
b
log_std/readIdentitylog_std*
_class
loc:@log_std*
T0*
_output_shapes
:
=
ExpExplog_std/read*
T0*
_output_shapes
:
d
random_normal/shapeConst*
dtype0*
valueB"и     *
_output_shapes
:
Я
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	и
k
random_normal/mulMul"random_normal/RandomStandardNormalExp*
T0*
_output_shapes
:	и
{
random_normalAddrandom_normal/mul(policy_network/fully_connected_1/BiasAdd*
T0*
_output_shapes
:	и
?
Exp_1Explog_std/read*
T0*
_output_shapes
:
§
bMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
©
fMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag_1/batch_shape_tensor/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
†
8MultivariateNormalDiag_1/shapes_from_loc_and_scale/ShapeShape(policy_network/fully_connected_1/BiasAdd*
out_type0*
T0*
_output_shapes
:
Р
FMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ы
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_1Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Т
HMultivariateNormalDiag_1/shapes_from_loc_and_scale/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ь
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
Њ
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
 *  А?*
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
А
>affine_linear_operator/init/DistributionShape/init/batch_ndimsConst*
dtype0*
value	B : *
_output_shapes
: 
А
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
Б
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
Г
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
valueB:*
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
Г
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
З
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
¬
MultivariateNormalDiag_2/rangeRangeMultivariateNormalDiag_2/sub$MultivariateNormalDiag_2/range/limit$MultivariateNormalDiag_2/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Є
DMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/subSuba_cont(policy_network/fully_connected_1/BiasAdd*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
У
–MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
ю
їMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
Џ
ЧMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
Э
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
џ
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
ф
ЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAddШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xЧMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
ъ
ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPackЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
_output_shapes
:*

axis *
T0*
N
Ю
ЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePackїMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
_output_shapes
:*

axis *
T0*
N
∞
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSliceШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
в
ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
Я
ЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1ShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
Ё
ЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/xConst*
dtype0*
value	B : *
_output_shapes
: 
Ь
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1AddЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/xїMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
T0*
_output_shapes
: 
Ь
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2AddШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1>affine_linear_operator/init/DistributionShape/init/batch_ndims*
T0*
_output_shapes
: 
ю
†MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/beginPackШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2*
_output_shapes
:*

axis *
T0*
N
Ґ
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/sizePack>affine_linear_operator/init/DistributionShape/init/event_ndims*
_output_shapes
:*

axis *
T0*
N
Є
ЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1SliceЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1†MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/beginЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/size*
Index0*
T0*
_output_shapes
:
 
ЗMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/pick_vector/condConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
Џ
ЖMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
≈
ВMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
П
}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concatConcatV2ЖMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/values_0ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeЪMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1ВMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
Ч
~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/ReshapeReshapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub}MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/concat*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ў
НMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ѕ
ЄMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
и
ЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
А
СMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose~MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/ReshapeЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
®
cMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
э
aMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivRealDivcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/xExp_1*
T0*
_output_shapes
:
≥
hMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
ф
dMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims
ExpandDimsaMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truedivhMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
Щ
]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mulMulСMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
„
ФMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/shiftConst*
dtype0*
value	B :*
_output_shapes
: 
»
њMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/assert_integer/statically_determined_was_integerNoOp
п
ЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
н
ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose	Transpose]MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mulЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ъ
„MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/DistributionShape/get_shape/ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
Е
¬MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndimsConst*
dtype0*
value	B :*
_output_shapes
: 
б
ЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zeroConst*
dtype0*
value	B : *
_output_shapes
: 
щ
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/ShapeShapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
в
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xConst*
dtype0*
value	B : *
_output_shapes
: 
Й
ЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/addAddЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add/xЮMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/zero*
T0*
_output_shapes
: 
И
•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/beginPackЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add*
_output_shapes
:*

axis *
T0*
N
ђ
§MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/sizePack¬MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
_output_shapes
:*

axis *
T0*
N
ћ
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/SliceSliceЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/begin§MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice/size*
Index0*
T0*
_output_shapes
:
й
•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shapeConst*
dtype0*
valueB *
_output_shapes
: 
ы
°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1ShapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
д
°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/xConst*
dtype0*
value	B : *
_output_shapes
: 
±
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1Add°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1/x¬MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/DistributionShape/get_shape/sample_ndims*
T0*
_output_shapes
: 
™
ЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2AddЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_1>affine_linear_operator/init/DistributionShape/init/batch_ndims*
T0*
_output_shapes
: 
М
ІMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/beginPackЯMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/add_2*
_output_shapes
:*

axis *
T0*
N
©
¶MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/sizePack>affine_linear_operator/init/DistributionShape/init/event_ndims*
_output_shapes
:*

axis *
T0*
N
‘
°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1Slice°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Shape_1ІMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/begin¶MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1/size*
Index0*
T0*
_output_shapes
:
ћ
ЙMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Њ
ДMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concatConcatV2ШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice•MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/batch_shape°MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/DistributionShape/get_shape/Slice_1ЙMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
ь
ЕMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/ReshapeReshapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transposeДMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/concat*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
©
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsAbsExp_1*
T0*
_output_shapes
:
Ъ
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogLogvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs*
T0*
_output_shapes
:
№
ИMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Њ
vMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/SumSumvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogИMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
ч
WMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/NegNegvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum*
T0*
_output_shapes
: 
Ч
AMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subSubЕMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape
Normal/loc*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
№
EMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truedivRealDivAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/subNormal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ƒ
8MultivariateNormalDiag_3/log_prob/Normal/log_prob/SquareSquareEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/xConst*
dtype0*
valueB
 *   њ*
_output_shapes
: 
к
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mulMul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
k
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/LogLogNormal/scale*
T0*
_output_shapes
: 
|
7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/xConst*
dtype0*
valueB
 *О?k?*
_output_shapes
: 
Ќ
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/addAdd7MultivariateNormalDiag_3/log_prob/Normal/log_prob/add/x5MultivariateNormalDiag_3/log_prob/Normal/log_prob/Log*
T0*
_output_shapes
: 
е
5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subSub5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul5MultivariateNormalDiag_3/log_prob/Normal/log_prob/add*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
√
%MultivariateNormalDiag_3/log_prob/SumSum5MultivariateNormalDiag_3/log_prob/Normal/log_prob/subMultivariateNormalDiag_2/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Џ
%MultivariateNormalDiag_3/log_prob/addAdd%MultivariateNormalDiag_3/log_prob/SumWMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg*
T0*#
_output_shapes
:€€€€€€€€€
_
NegNeg%MultivariateNormalDiag_3/log_prob/add*
T0*#
_output_shapes
:€€€€€€€€€
D
mulMulNegadvtg*
T0*#
_output_shapes
:€€€€€€€€€
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
 *  А?*
_output_shapes
: 
j
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*#
_output_shapes
:€€€€€€€€€
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
і
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
b
gradients/mul_grad/mulMulgradients/Filladvtg*
T0*#
_output_shapes
:€€€€€€€€€
Я
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
У
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
b
gradients/mul_grad/mul_1MulNeggradients/Fill*
T0*#
_output_shapes
:€€€€€€€€€
•
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Щ
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
÷
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*#
_output_shapes
:€€€€€€€€€
№
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*#
_output_shapes
:€€€€€€€€€
x
gradients/Neg_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
T0*#
_output_shapes
:€€€€€€€€€
®
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/ShapeShape%MultivariateNormalDiag_3/log_prob/Sum*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€

<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
Ъ
Jgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
г
8gradients/MultivariateNormalDiag_3/log_prob/add_grad/SumSumgradients/Neg_grad/NegJgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
о
<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeReshape8gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:
з
:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1Sumgradients/Neg_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
т
>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1Reshape:gradients/MultivariateNormalDiag_3/log_prob/add_grad/Sum_1<gradients/MultivariateNormalDiag_3/log_prob/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ќ
Egradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_depsNoOp=^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape?^gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1
”
Mgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyIdentity<gradients/MultivariateNormalDiag_3/log_prob/add_grad/ReshapeF^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*O
_classE
CAloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape*
T0*
_output_shapes
:
„
Ogradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1Identity>gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1F^gradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/MultivariateNormalDiag_3/log_prob/add_grad/Reshape_1*
T0*
_output_shapes
: 
ѓ
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub*
out_type0*
T0*
_output_shapes
:
 
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/SizeConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ч
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/addAddMultivariateNormalDiag_2/range9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ґ
8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/modFloorMod8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/add9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Г
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1Shape8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
:
—
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/startConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B : *
_output_shapes
: 
—
@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/deltaConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
щ
:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/rangeRange@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/start9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Size@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range/delta*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*

Tidx0*
_output_shapes
:
–
?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/valueConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
љ
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/FillFill<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape_1?gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill/value*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
∆
Bgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitchDynamicStitch:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/range8gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/mod:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Fill*
N*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ѕ
>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/yConst*
dtype0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
value	B :*
_output_shapes
: 
»
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/MaximumMaximumBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch>gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum/y*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Ј
=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordivFloorDiv:gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Maximum*M
_classC
A?loc:@gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Shape*
T0*
_output_shapes
:
Л
<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/ReshapeReshapeMgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependencyBgradients/MultivariateNormalDiag_3/log_prob/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
Л
9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileTile<gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Reshape=gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
г
jgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/NegNegOgradients/MultivariateNormalDiag_3/log_prob/add_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
њ
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeShape5MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul*
out_type0*
T0*
_output_shapes
:
П
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
 
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
¶
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumSum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/TileZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ґ
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
™
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1Sum9gradients/MultivariateNormalDiag_3/log_prob/Sum_grad/Tile\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Њ
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegNegJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
†
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1ReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/NegLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
э
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1
Ђ
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ч
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/Reshape_1*
T0*
_output_shapes
: 
÷
Лgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
с
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/SizeConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
т
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/addAddИMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
ш
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/modFloorModЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/addКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Size*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
ь
Нgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1Const*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
valueB:*
_output_shapes
:
ш
Сgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/startConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B : *
_output_shapes
: 
ш
Сgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/deltaConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ц
Лgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/rangeRangeСgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/startКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/SizeСgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/range/delta*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*

Tidx0*
_output_shapes
:
ч
Рgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/valueConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
€
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/FillFillНgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape_1Рgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill/value*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*
_output_shapes
:
µ
Уgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitchDynamicStitchЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/rangeЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/modЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Fill*
N*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ц
Пgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/yConst*
dtype0*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
value	B :*
_output_shapes
: 
У
Нgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/MaximumMaximumУgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitchПgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum/y*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Л
Оgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordivFloorDivЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ShapeНgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Maximum*°
_classЦ
УРloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ћ
Нgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ReshapeReshapejgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg_grad/NegУgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
л
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/TileTileНgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/ReshapeОgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
Н
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
ƒ
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1Shape8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
out_type0*
T0*
_output_shapes
:
 
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ShapeLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
£
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulMul]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency8MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
µ
Hgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumSumHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mulZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ь
Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeReshapeHgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
§
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1Mul7MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul/x]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/sub_grad/tuple/control_dependency*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ї
Jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1SumJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/mul_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Љ
Ngradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1ReshapeJgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Sum_1Lgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Shape_1*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
э
Ugradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_depsNoOpM^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeO^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1
С
]gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependencyIdentityLgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/ReshapeV^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape*
T0*
_output_shapes
: 
±
_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1IdentityNgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1V^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/Reshape_1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
Рgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal
ReciprocalvMultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsЛ^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/Tile*
T0*
_output_shapes
:
÷
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mulMulКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum_grad/TileРgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/Reciprocal*
T0*
_output_shapes
:
ф
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xConst`^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1*
dtype0*
valueB
 *   @*
_output_shapes
: 
£
Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul/xEMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
љ
Mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Mul_gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/mul_grad/tuple/control_dependency_1Kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
њ
Кgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/SignSignExp_1*
T0*
_output_shapes
:
ѕ
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mulMulЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log_grad/mulКgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/Sign*
T0*
_output_shapes
:
џ
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ShapeShapeAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
out_type0*
T0*
_output_shapes
:
Я
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
ъ
jgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivRealDivMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1Normal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
й
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumSum\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDivjgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ж
\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/ReshapeReshapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/SumZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ё
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNegAMultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
М
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1RealDivXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/NegNormal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Т
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2RealDiv^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_1Normal/scale*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
…
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mulMulMgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/Square_grad/mul_1^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/RealDiv_2*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
й
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1SumXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/mullgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
“
^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1ReshapeZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Sum_1\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
≠
egradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_depsNoOp]^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1
л
mgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentity\gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshapef^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*o
_classe
caloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
„
ogradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependency_1Identity^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1f^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/group_deps*q
_classg
ecloc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
Ь
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeShapeЕMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape*
out_type0*
T0*
_output_shapes
:
Ы
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
о
fgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ShapeXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumSummgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyfgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Џ
Xgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/ReshapeReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/SumVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ц
Vgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1Summgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/truediv_grad/tuple/control_dependencyhgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
÷
Tgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegNegVgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
ƒ
Zgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1ReshapeTgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/NegXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
°
agradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_depsNoOpY^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape[^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1
џ
igradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependencyIdentityXgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshapeb^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*k
_classa
_]loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
«
kgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependency_1IdentityZgradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1b^gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/Reshape_1*
T0*
_output_shapes
: 
ф
Ъgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ShapeShapeШMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
щ
Ьgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshapeigradients/MultivariateNormalDiag_3/log_prob/Normal/log_prob/standardize/sub_grad/tuple/control_dependencyЪgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ф
єgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutationЭMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
в
±gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	TransposeЬgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/Reshape_grad/Reshapeєgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ƒ
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ShapeShapeСMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose*
out_type0*
T0*
_output_shapes
:
≈
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
√
Вgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgsBroadcastGradientArgsrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mulMul±gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transposedMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ѓ
pgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/SumSumpgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mulВgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/ReshapeReshapepgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ь
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1MulСMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose±gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape_1/undo_make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
і
rgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1Sumrgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/mul_1Дgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ґ
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1Reshapergradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Sum_1tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:
х
}gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_depsNoOpu^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshapew^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1
Ќ
Еgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependencyIdentitytgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*З
_class}
{yloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ѕ
Зgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1Identityvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1~^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*Й
_class
}{loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/Reshape_1*
T0*
_output_shapes

:
Ж
≤gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutationInvertPermutationЦMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose/perm*
T0*
_output_shapes
:
љ
™gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transpose	TransposeЕgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency≤gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/InvertPermutation*
Tperm0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
√
ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Њ
{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeReshapeЗgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Shape*
Tshape0*
T0*
_output_shapes
:
Ш
Уgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ShapeShapeDMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
≠
Хgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/ReshapeReshape™gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/rotate_transpose/transpose_grad/transposeУgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
є
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
¬
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
ѕ
Жgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shapexgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ђ
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDivRealDiv{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeExp_1*
T0*
_output_shapes
:
Њ
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/SumSumxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDivЖgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
†
xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ReshapeReshapetgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sumvgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Б
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegNegcMultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv/x*
T0*
_output_shapes
: 
І
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1RealDivtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/NegExp_1*
T0*
_output_shapes
:
≠
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2RealDivzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_1Exp_1*
T0*
_output_shapes
:
Щ
tgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mulMul{gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Reshapezgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/RealDiv_2*
T0*
_output_shapes
:
Њ
vgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1Sumtgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/mulИgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
™
zgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1Reshapevgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Sum_1xgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
В
Бgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_depsNoOpy^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape{^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1
≈
Йgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependencyIdentityxgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/ReshapeВ^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*М
_classБ
}loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape*
T0*
_output_shapes
: 
–
Лgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1Identityzgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1В^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/group_deps*П
_classД
Бloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/Reshape_1*
T0*
_output_shapes
:
А
gradients/AddNAddNЙgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mulЛgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/truediv_grad/tuple/control_dependency_1*
N*Я
_classФ
СОloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs_grad/mul*
T0*
_output_shapes
:
[
gradients/Exp_1_grad/mulMulgradients/AddNExp_1*
T0*
_output_shapes
:
Я
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ShapeShapea_cont*
out_type0*
T0*
_output_shapes
:
√
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1Shape(policy_network/fully_connected_1/BiasAdd*
out_type0*
T0*
_output_shapes
:
ч
igradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgsBroadcastGradientArgsYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
°
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumSumХgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapeigradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
г
[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/ReshapeReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/SumYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
•
Ygradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1SumХgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/DistributionShape/make_batch_of_event_sample_matrices/Reshape_grad/Reshapekgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
№
Wgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/NegNegYgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1*
T0*
_output_shapes
:
ё
]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1ReshapeWgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Neg[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
™
dgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_depsNoOp\^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape^^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1
з
lgradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependencyIdentity[gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshapee^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*n
_classd
b`loc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
д
ngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1Identity]gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1e^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
ю
Cgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGradngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
T0*
_output_shapes
:
З
Hgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOpo^gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1D^gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad
ї
Pgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentityngradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1I^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*p
_classf
dbloc:@gradients/MultivariateNormalDiag_3/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
л
Rgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGradI^gradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/policy_network/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
†
=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulMatMulPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency-policy_network/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
П
?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1MatMul#policy_network/fully_connected/ReluPgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
—
Ggradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_depsNoOp>^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul@^gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1
и
Ogradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity=gradients/policy_network/fully_connected_1/MatMul_grad/MatMulH^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
е
Qgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity?gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1H^gradients/policy_network/fully_connected_1/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/policy_network/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
п
;gradients/policy_network/fully_connected/Relu_grad/ReluGradReluGradOgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency#policy_network/fully_connected/Relu*
T0*'
_output_shapes
:€€€€€€€€€
…
Agradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
–
Fgradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/Relu_grad/ReluGradB^gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad
в
Ngradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/Relu_grad/ReluGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:€€€€€€€€€
г
Pgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGradG^gradients/policy_network/fully_connected/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/policy_network/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
Ъ
;gradients/policy_network/fully_connected/MatMul_grad/MatMulMatMulNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency+policy_network/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
й
=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1MatMuloNgradients/policy_network/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
Ћ
Egradients/policy_network/fully_connected/MatMul_grad/tuple/group_depsNoOp<^gradients/policy_network/fully_connected/MatMul_grad/MatMul>^gradients/policy_network/fully_connected/MatMul_grad/MatMul_1
а
Mgradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependencyIdentity;gradients/policy_network/fully_connected/MatMul_grad/MatMulF^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
Ё
Ogradients/policy_network/fully_connected/MatMul_grad/tuple/control_dependency_1Identity=gradients/policy_network/fully_connected/MatMul_grad/MatMul_1F^gradients/policy_network/fully_connected/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@gradients/policy_network/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
z
beta1_power/initial_valueConst*
dtype0*
_class
loc:@log_std*
valueB
 *fff?*
_output_shapes
: 
Л
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
™
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
 *wЊ?*
_output_shapes
: 
Л
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
™
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
Ќ
=policy_network/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
Џ
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
µ
2policy_network/fully_connected/weights/Adam/AssignAssign+policy_network/fully_connected/weights/Adam=policy_network/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
Ќ
0policy_network/fully_connected/weights/Adam/readIdentity+policy_network/fully_connected/weights/Adam*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
ѕ
?policy_network/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@policy_network/fully_connected/weights*
valueB*    *
_output_shapes

:
№
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
ї
4policy_network/fully_connected/weights/Adam_1/AssignAssign-policy_network/fully_connected/weights/Adam_1?policy_network/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*9
_class/
-+loc:@policy_network/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
—
2policy_network/fully_connected/weights/Adam_1/readIdentity-policy_network/fully_connected/weights/Adam_1*9
_class/
-+loc:@policy_network/fully_connected/weights*
T0*
_output_shapes

:
√
<policy_network/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
–
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
≠
1policy_network/fully_connected/biases/Adam/AssignAssign*policy_network/fully_connected/biases/Adam<policy_network/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
∆
/policy_network/fully_connected/biases/Adam/readIdentity*policy_network/fully_connected/biases/Adam*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
≈
>policy_network/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@policy_network/fully_connected/biases*
valueB*    *
_output_shapes
:
“
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
≥
3policy_network/fully_connected/biases/Adam_1/AssignAssign,policy_network/fully_connected/biases/Adam_1>policy_network/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@policy_network/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
 
1policy_network/fully_connected/biases/Adam_1/readIdentity,policy_network/fully_connected/biases/Adam_1*8
_class.
,*loc:@policy_network/fully_connected/biases*
T0*
_output_shapes
:
—
?policy_network/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
ё
-policy_network/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
љ
4policy_network/fully_connected_1/weights/Adam/AssignAssign-policy_network/fully_connected_1/weights/Adam?policy_network/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
”
2policy_network/fully_connected_1/weights/Adam/readIdentity-policy_network/fully_connected_1/weights/Adam*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
”
Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
valueB*    *
_output_shapes

:
а
/policy_network/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
shared_name 
√
6policy_network/fully_connected_1/weights/Adam_1/AssignAssign/policy_network/fully_connected_1/weights/Adam_1Apolicy_network/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
„
4policy_network/fully_connected_1/weights/Adam_1/readIdentity/policy_network/fully_connected_1/weights/Adam_1*;
_class1
/-loc:@policy_network/fully_connected_1/weights*
T0*
_output_shapes

:
«
>policy_network/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
‘
,policy_network/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
µ
3policy_network/fully_connected_1/biases/Adam/AssignAssign,policy_network/fully_connected_1/biases/Adam>policy_network/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
ћ
1policy_network/fully_connected_1/biases/Adam/readIdentity,policy_network/fully_connected_1/biases/Adam*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
…
@policy_network/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
valueB*    *
_output_shapes
:
÷
.policy_network/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
shared_name 
ї
5policy_network/fully_connected_1/biases/Adam_1/AssignAssign.policy_network/fully_connected_1/biases/Adam_1@policy_network/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
–
3policy_network/fully_connected_1/biases/Adam_1/readIdentity.policy_network/fully_connected_1/biases/Adam_1*:
_class0
.,loc:@policy_network/fully_connected_1/biases*
T0*
_output_shapes
:
З
log_std/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
Ф
log_std/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
µ
log_std/Adam/AssignAssignlog_std/Adamlog_std/Adam/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
l
log_std/Adam/readIdentitylog_std/Adam*
_class
loc:@log_std*
T0*
_output_shapes
:
Й
 log_std/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@log_std*
valueB*    *
_output_shapes
:
Ц
log_std/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@log_std*
shared_name 
ї
log_std/Adam_1/AssignAssignlog_std/Adam_1 log_std/Adam_1/Initializer/zeros*
validate_shape(*
_class
loc:@log_std*
use_locking(*
T0*
_output_shapes
:
p
log_std/Adam_1/readIdentitylog_std/Adam_1*
_class
loc:@log_std*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *П¬х<*
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
 *wЊ?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
З
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
€
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
У
>Adam/update_policy_network/fully_connected_1/weights/ApplyAdam	ApplyAdam(policy_network/fully_connected_1/weights-policy_network/fully_connected_1/weights/Adam/policy_network/fully_connected_1/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonQgradients/policy_network/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *;
_class1
/-loc:@policy_network/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:
Л
=Adam/update_policy_network/fully_connected_1/biases/ApplyAdam	ApplyAdam'policy_network/fully_connected_1/biases,policy_network/fully_connected_1/biases/Adam.policy_network/fully_connected_1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonRgradients/policy_network/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *:
_class0
.,loc:@policy_network/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
:
±
Adam/update_log_std/ApplyAdam	ApplyAdamlog_stdlog_std/Adamlog_std/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/Exp_1_grad/mul*
use_nesterov( *
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
:
И
Adam/mulMulbeta1_power/read
Adam/beta1=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
Т
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
К

Adam/mul_1Mulbeta2_power/read
Adam/beta2=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/update_log_std/ApplyAdam*
_class
loc:@log_std*
T0*
_output_shapes
: 
Ц
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_class
loc:@log_std*
use_locking( *
T0*
_output_shapes
: 
»
AdamNoOp=^Adam/update_policy_network/fully_connected/weights/ApplyAdam<^Adam/update_policy_network/fully_connected/biases/ApplyAdam?^Adam/update_policy_network/fully_connected_1/weights/ApplyAdam>^Adam/update_policy_network/fully_connected_1/biases/ApplyAdam^Adam/update_log_std/ApplyAdam^Adam/Assign^Adam/Assign_1
«
Abaseline/fully_connected/weights/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB"      *
_output_shapes
:
є
?baseline/fully_connected/weights/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *М7њ*
_output_shapes
: 
є
?baseline/fully_connected/weights/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB
 *М7?*
_output_shapes
: 
°
Ibaseline/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformAbaseline/fully_connected/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@baseline/fully_connected/weights
Ю
?baseline/fully_connected/weights/Initializer/random_uniform/subSub?baseline/fully_connected/weights/Initializer/random_uniform/max?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes
: 
∞
?baseline/fully_connected/weights/Initializer/random_uniform/mulMulIbaseline/fully_connected/weights/Initializer/random_uniform/RandomUniform?baseline/fully_connected/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
Ґ
;baseline/fully_connected/weights/Initializer/random_uniformAdd?baseline/fully_connected/weights/Initializer/random_uniform/mul?baseline/fully_connected/weights/Initializer/random_uniform/min*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
…
 baseline/fully_connected/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
Ч
'baseline/fully_connected/weights/AssignAssign baseline/fully_connected/weights;baseline/fully_connected/weights/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
±
%baseline/fully_connected/weights/readIdentity baseline/fully_connected/weights*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
≤
1baseline/fully_connected/biases/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB*    *
_output_shapes
:
њ
baseline/fully_connected/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
Ж
&baseline/fully_connected/biases/AssignAssignbaseline/fully_connected/biases1baseline/fully_connected/biases/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
™
$baseline/fully_connected/biases/readIdentitybaseline/fully_connected/biases*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
:
Ђ
baseline/fully_connected/MatMulMatMulo%baseline/fully_connected/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
ї
 baseline/fully_connected/BiasAddBiasAddbaseline/fully_connected/MatMul$baseline/fully_connected/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€
y
baseline/fully_connected/ReluRelu baseline/fully_connected/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Ћ
Cbaseline/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB"      *
_output_shapes
:
љ
Abaseline/fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *0њ*
_output_shapes
: 
љ
Abaseline/fully_connected_1/weights/Initializer/random_uniform/maxConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB
 *0?*
_output_shapes
: 
І
Kbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformCbaseline/fully_connected_1/weights/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*5
_class+
)'loc:@baseline/fully_connected_1/weights
¶
Abaseline/fully_connected_1/weights/Initializer/random_uniform/subSubAbaseline/fully_connected_1/weights/Initializer/random_uniform/maxAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes
: 
Є
Abaseline/fully_connected_1/weights/Initializer/random_uniform/mulMulKbaseline/fully_connected_1/weights/Initializer/random_uniform/RandomUniformAbaseline/fully_connected_1/weights/Initializer/random_uniform/sub*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
™
=baseline/fully_connected_1/weights/Initializer/random_uniformAddAbaseline/fully_connected_1/weights/Initializer/random_uniform/mulAbaseline/fully_connected_1/weights/Initializer/random_uniform/min*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
Ќ
"baseline/fully_connected_1/weights
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
Я
)baseline/fully_connected_1/weights/AssignAssign"baseline/fully_connected_1/weights=baseline/fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
Ј
'baseline/fully_connected_1/weights/readIdentity"baseline/fully_connected_1/weights*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
ґ
3baseline/fully_connected_1/biases/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB*    *
_output_shapes
:
√
!baseline/fully_connected_1/biases
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
О
(baseline/fully_connected_1/biases/AssignAssign!baseline/fully_connected_1/biases3baseline/fully_connected_1/biases/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
∞
&baseline/fully_connected_1/biases/readIdentity!baseline/fully_connected_1/biases*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
:
Ћ
!baseline/fully_connected_1/MatMulMatMulbaseline/fully_connected/Relu'baseline/fully_connected_1/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
Ѕ
"baseline/fully_connected_1/BiasAddBiasAdd!baseline/fully_connected_1/MatMul&baseline/fully_connected_1/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€
e

baseline_1Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
m
SqueezeSqueeze"baseline/fully_connected_1/BiasAdd*
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
 *  А?*
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
°
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
В
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
∆
1mean_squared_error/assert_broadcastable/is_scalarEqual3mean_squared_error/assert_broadcastable/is_scalar/x4mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
–
=mean_squared_error/assert_broadcastable/is_valid_shape/SwitchSwitch1mean_squared_error/assert_broadcastable/is_scalar1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
≠
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_tIdentity?mean_squared_error/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
Ђ
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_fIdentity=mean_squared_error/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
Ю
>mean_squared_error/assert_broadcastable/is_valid_shape/pred_idIdentity1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
•
?mean_squared_error/assert_broadcastable/is_valid_shape/Switch_1Switch1mean_squared_error/assert_broadcastable/is_scalar>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
’
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualdmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchfmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
ќ
dmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch3mean_squared_error/assert_broadcastable/values/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*F
_class<
:8loc:@mean_squared_error/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
“
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch4mean_squared_error/assert_broadcastable/weights/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
¬
Wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
б
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
я
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityWmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
д
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Ч
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
£
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimswmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
щ
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch4mean_squared_error/assert_broadcastable/values/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
‘
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchsmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapelmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
П
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
Л
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:€€€€€€€€€
Л
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
Е
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimskmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0*
N
Щ
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ю
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
з
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch5mean_squared_error/assert_broadcastable/weights/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
√
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchumean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
з
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationnmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
set_operationa-b*
T0*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:
©
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
Б
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
д
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualcmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xrmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
±
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*p
_classf
dbloc:@mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
…
Vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
М
<mean_squared_error/assert_broadcastable/is_valid_shape/MergeMergeVmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeAmean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
Х
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
°
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
Ц
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
г
:mean_squared_error/assert_broadcastable/AssertGuard/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
І
<mean_squared_error/assert_broadcastable/AssertGuard/switch_tIdentity<mean_squared_error/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
•
<mean_squared_error/assert_broadcastable/AssertGuard/switch_fIdentity:mean_squared_error/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
¶
;mean_squared_error/assert_broadcastable/AssertGuard/pred_idIdentity<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 

8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp=^mean_squared_error/assert_broadcastable/AssertGuard/switch_t
љ
Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependencyIdentity<mean_squared_error/assert_broadcastable/AssertGuard/switch_t9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
и
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
ѕ
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
т
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
_output_shapes
: 
ќ
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
з
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
Ћ
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
Н
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertAmean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchAmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

Ї
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
≤
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Switch5mean_squared_error/assert_broadcastable/weights/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
∆
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Switch4mean_squared_error/assert_broadcastable/values/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
¶
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3Switch1mean_squared_error/assert_broadcastable/is_scalar;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
Ѕ
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Identity<mean_squared_error/assert_broadcastable/AssertGuard/switch_f;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
А
9mean_squared_error/assert_broadcastable/AssertGuard/MergeMergeHmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
Я
mean_squared_error/ToFloat_3/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Ж
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
X
mean_squared_error/RankRankmean_squared_error/Mul*
T0*
_output_shapes
: 
Ь
mean_squared_error/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
Ь
mean_squared_error/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
Ђ
mean_squared_error/rangeRangemean_squared_error/range/startmean_squared_error/Rankmean_squared_error/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Н
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
І
&mean_squared_error/num_present/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
Ц
$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat_3/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
™
)mean_squared_error/num_present/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
≠
.mean_squared_error/num_present/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
ѓ
.mean_squared_error/num_present/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
±
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*
_output_shapes
: 
Ћ
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
“
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
–
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
ы
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
№
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
: 
ѕ
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
†
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
™
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
й
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
з
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
Џ
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Э
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
±
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualВmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchДmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
«
Вmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*d
_classZ
XVloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank*
T0*
_output_shapes
: : 
Ћ
Дmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: : 
Ь
umean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Э
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Ы
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityumean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
†
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Р
Оmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
А
Кmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsХmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Оmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
т
Сmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchСmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
џ
Пmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapeКmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
out_type0*
T0*
_output_shapes
:
И
Пmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
и
Йmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillПmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeПmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:€€€€€€€€€
Д
Лmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B :*
_output_shapes
: 
Б
Жmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Кmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsЙmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeЛmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0*
N
Т
Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
ы
Мmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЧmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
а
Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
љ
Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchУmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
ƒ
Шmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationМmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Жmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
set_operationa-b*
T0*<
_output_shapes*
(:€€€€€€€€€:€€€€€€€€€:
з
Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeЪmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
T0*
_output_shapes
: 
ъ
Бmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
value	B : *
_output_shapes
: 
ј
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualБmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xРmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
≠
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*С
_classЖ
ГАloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
£
tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergewmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
ж
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergetmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
п
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
Ў
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
с
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
„
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
р
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
‘
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
љ
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
г
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
б
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
в
Ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
ч
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
µ
dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
T0
*
_output_shapes
: 
а
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'weights can not be broadcast to values.*
_output_shapes
: 
«
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bweights.shape=*
_output_shapes
: 
а
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0*
_output_shapes
: 
∆
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB Bvalues.shape=*
_output_shapes
: 
я
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0*
_output_shapes
: 
√
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=*
_output_shapes
: 
„
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

≤
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
™
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
T0*
_output_shapes

: : 
Њ
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ю
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
є
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
T0
*
_output_shapes
: 
Џ
Wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergefmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
√
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
Ы
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
й
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*
_output_shapes
:
љ
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
~
#mean_squared_error/num_present/RankRank0mean_squared_error/num_present/broadcast_weights*
T0*
_output_shapes
: 
®
*mean_squared_error/num_present/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B : *
_output_shapes
: 
®
*mean_squared_error/num_present/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
value	B :*
_output_shapes
: 
џ
$mean_squared_error/num_present/rangeRange*mean_squared_error/num_present/range/start#mean_squared_error/num_present/Rank*mean_squared_error/num_present/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
ї
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ч
mean_squared_error/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
П
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Э
mean_squared_error/Greater/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
Д
mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
T0*
_output_shapes
: 
Ы
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
°
"mean_squared_error/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB *
_output_shapes
: 
£
"mean_squared_error/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Н
mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*
_output_shapes
: 
Ь
mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ю
mean_squared_error/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
valueB
 *    *
_output_shapes
: 
Ц
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
 *  А?*
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
њ
0gradients_1/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients_1/Fill4gradients_1/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
Ѕ
2gradients_1/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater4gradients_1/mean_squared_error/value_grad/zeros_likegradients_1/Fill*
T0*
_output_shapes
: 
™
:gradients_1/mean_squared_error/value_grad/tuple/group_depsNoOp1^gradients_1/mean_squared_error/value_grad/Select3^gradients_1/mean_squared_error/value_grad/Select_1
£
Bgradients_1/mean_squared_error/value_grad/tuple/control_dependencyIdentity0gradients_1/mean_squared_error/value_grad/Select;^gradients_1/mean_squared_error/value_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/mean_squared_error/value_grad/Select*
T0*
_output_shapes
: 
©
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
у
=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/div_grad/Shape/gradients_1/mean_squared_error/div_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ї
/gradients_1/mean_squared_error/div_grad/RealDivRealDivBgradients_1/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
в
+gradients_1/mean_squared_error/div_grad/SumSum/gradients_1/mean_squared_error/div_grad/RealDiv=gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
≈
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
•
1gradients_1/mean_squared_error/div_grad/RealDiv_1RealDiv+gradients_1/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
Ђ
1gradients_1/mean_squared_error/div_grad/RealDiv_2RealDiv1gradients_1/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
 
+gradients_1/mean_squared_error/div_grad/mulMulBgradients_1/mean_squared_error/value_grad/tuple/control_dependency1gradients_1/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
в
-gradients_1/mean_squared_error/div_grad/Sum_1Sum+gradients_1/mean_squared_error/div_grad/mul?gradients_1/mean_squared_error/div_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ћ
1gradients_1/mean_squared_error/div_grad/Reshape_1Reshape-gradients_1/mean_squared_error/div_grad/Sum_1/gradients_1/mean_squared_error/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
¶
8gradients_1/mean_squared_error/div_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/div_grad/Reshape2^gradients_1/mean_squared_error/div_grad/Reshape_1
Э
@gradients_1/mean_squared_error/div_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/div_grad/Reshape9^gradients_1/mean_squared_error/div_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/div_grad/Reshape*
T0*
_output_shapes
: 
£
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
ж
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
÷
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
с
1gradients_1/mean_squared_error/Select_grad/SelectSelectmean_squared_error/EqualBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_15gradients_1/mean_squared_error/Select_grad/zeros_like*
T0*
_output_shapes
: 
у
3gradients_1/mean_squared_error/Select_grad/Select_1Selectmean_squared_error/Equal5gradients_1/mean_squared_error/Select_grad/zeros_likeBgradients_1/mean_squared_error/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
≠
;gradients_1/mean_squared_error/Select_grad/tuple/group_depsNoOp2^gradients_1/mean_squared_error/Select_grad/Select4^gradients_1/mean_squared_error/Select_grad/Select_1
І
Cgradients_1/mean_squared_error/Select_grad/tuple/control_dependencyIdentity1gradients_1/mean_squared_error/Select_grad/Select<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Select_grad/Select*
T0*
_output_shapes
: 
≠
Egradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1Identity3gradients_1/mean_squared_error/Select_grad/Select_1<^gradients_1/mean_squared_error/Select_grad/tuple/group_deps*F
_class<
:8loc:@gradients_1/mean_squared_error/Select_grad/Select_1*
T0*
_output_shapes
: 
М
-gradients_1/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
÷
,gradients_1/mean_squared_error/Sum_grad/SizeSize-gradients_1/mean_squared_error/Sum_grad/Shape*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
: 
к
+gradients_1/mean_squared_error/Sum_grad/addAddmean_squared_error/range,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
В
+gradients_1/mean_squared_error/Sum_grad/modFloorMod+gradients_1/mean_squared_error/Sum_grad/add,gradients_1/mean_squared_error/Sum_grad/Size*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
№
/gradients_1/mean_squared_error/Sum_grad/Shape_1Shape+gradients_1/mean_squared_error/Sum_grad/mod*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
out_type0*
T0*
_output_shapes
:
Ј
3gradients_1/mean_squared_error/Sum_grad/range/startConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B : *
_output_shapes
: 
Ј
3gradients_1/mean_squared_error/Sum_grad/range/deltaConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ѕ
-gradients_1/mean_squared_error/Sum_grad/rangeRange3gradients_1/mean_squared_error/Sum_grad/range/start,gradients_1/mean_squared_error/Sum_grad/Size3gradients_1/mean_squared_error/Sum_grad/range/delta*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*

Tidx0*#
_output_shapes
:€€€€€€€€€
ґ
2gradients_1/mean_squared_error/Sum_grad/Fill/valueConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Й
,gradients_1/mean_squared_error/Sum_grad/FillFill/gradients_1/mean_squared_error/Sum_grad/Shape_12gradients_1/mean_squared_error/Sum_grad/Fill/value*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ш
5gradients_1/mean_squared_error/Sum_grad/DynamicStitchDynamicStitch-gradients_1/mean_squared_error/Sum_grad/range+gradients_1/mean_squared_error/Sum_grad/mod-gradients_1/mean_squared_error/Sum_grad/Shape,gradients_1/mean_squared_error/Sum_grad/Fill*
N*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
µ
1gradients_1/mean_squared_error/Sum_grad/Maximum/yConst*
dtype0*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ф
/gradients_1/mean_squared_error/Sum_grad/MaximumMaximum5gradients_1/mean_squared_error/Sum_grad/DynamicStitch1gradients_1/mean_squared_error/Sum_grad/Maximum/y*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
М
0gradients_1/mean_squared_error/Sum_grad/floordivFloorDiv-gradients_1/mean_squared_error/Sum_grad/Shape/gradients_1/mean_squared_error/Sum_grad/Maximum*@
_class6
42loc:@gradients_1/mean_squared_error/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
“
/gradients_1/mean_squared_error/Sum_grad/ReshapeReshape.gradients_1/mean_squared_error/Sum_1_grad/Tile5gradients_1/mean_squared_error/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
ћ
,gradients_1/mean_squared_error/Sum_grad/TileTile/gradients_1/mean_squared_error/Sum_grad/Reshape0gradients_1/mean_squared_error/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
Ѓ
5gradients_1/mean_squared_error/num_present_grad/ShapeShape0mean_squared_error/num_present/broadcast_weights*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
о
4gradients_1/mean_squared_error/num_present_grad/SizeSize5gradients_1/mean_squared_error/num_present_grad/Shape*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
out_type0*
T0*
_output_shapes
: 
О
3gradients_1/mean_squared_error/num_present_grad/addAdd$mean_squared_error/num_present/range4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Ґ
3gradients_1/mean_squared_error/num_present_grad/modFloorMod3gradients_1/mean_squared_error/num_present_grad/add4gradients_1/mean_squared_error/num_present_grad/Size*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ф
7gradients_1/mean_squared_error/num_present_grad/Shape_1Shape3gradients_1/mean_squared_error/num_present_grad/mod*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
out_type0*
T0*
_output_shapes
:
«
;gradients_1/mean_squared_error/num_present_grad/range/startConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B : *
_output_shapes
: 
«
;gradients_1/mean_squared_error/num_present_grad/range/deltaConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
й
5gradients_1/mean_squared_error/num_present_grad/rangeRange;gradients_1/mean_squared_error/num_present_grad/range/start4gradients_1/mean_squared_error/num_present_grad/Size;gradients_1/mean_squared_error/num_present_grad/range/delta*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*

Tidx0*#
_output_shapes
:€€€€€€€€€
∆
:gradients_1/mean_squared_error/num_present_grad/Fill/valueConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
©
4gradients_1/mean_squared_error/num_present_grad/FillFill7gradients_1/mean_squared_error/num_present_grad/Shape_1:gradients_1/mean_squared_error/num_present_grad/Fill/value*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
®
=gradients_1/mean_squared_error/num_present_grad/DynamicStitchDynamicStitch5gradients_1/mean_squared_error/num_present_grad/range3gradients_1/mean_squared_error/num_present_grad/mod5gradients_1/mean_squared_error/num_present_grad/Shape4gradients_1/mean_squared_error/num_present_grad/Fill*
N*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
≈
9gradients_1/mean_squared_error/num_present_grad/Maximum/yConst*
dtype0*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
value	B :*
_output_shapes
: 
і
7gradients_1/mean_squared_error/num_present_grad/MaximumMaximum=gradients_1/mean_squared_error/num_present_grad/DynamicStitch9gradients_1/mean_squared_error/num_present_grad/Maximum/y*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
ђ
8gradients_1/mean_squared_error/num_present_grad/floordivFloorDiv5gradients_1/mean_squared_error/num_present_grad/Shape7gradients_1/mean_squared_error/num_present_grad/Maximum*H
_class>
<:loc:@gradients_1/mean_squared_error/num_present_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
щ
7gradients_1/mean_squared_error/num_present_grad/ReshapeReshapeEgradients_1/mean_squared_error/Select_grad/tuple/control_dependency_1=gradients_1/mean_squared_error/num_present_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
д
4gradients_1/mean_squared_error/num_present_grad/TileTile7gradients_1/mean_squared_error/num_present_grad/Reshape8gradients_1/mean_squared_error/num_present_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
Ъ
-gradients_1/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
r
/gradients_1/mean_squared_error/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
у
=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/mean_squared_error/Mul_grad/Shape/gradients_1/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
£
+gradients_1/mean_squared_error/Mul_grad/mulMul,gradients_1/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat_3/x*
T0*
_output_shapes
:
ё
+gradients_1/mean_squared_error/Mul_grad/SumSum+gradients_1/mean_squared_error/Mul_grad/mul=gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
«
/gradients_1/mean_squared_error/Mul_grad/ReshapeReshape+gradients_1/mean_squared_error/Mul_grad/Sum-gradients_1/mean_squared_error/Mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
Ђ
-gradients_1/mean_squared_error/Mul_grad/mul_1Mul$mean_squared_error/SquaredDifference,gradients_1/mean_squared_error/Sum_grad/Tile*
T0*
_output_shapes
:
д
-gradients_1/mean_squared_error/Mul_grad/Sum_1Sum-gradients_1/mean_squared_error/Mul_grad/mul_1?gradients_1/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ћ
1gradients_1/mean_squared_error/Mul_grad/Reshape_1Reshape-gradients_1/mean_squared_error/Mul_grad/Sum_1/gradients_1/mean_squared_error/Mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
¶
8gradients_1/mean_squared_error/Mul_grad/tuple/group_depsNoOp0^gradients_1/mean_squared_error/Mul_grad/Reshape2^gradients_1/mean_squared_error/Mul_grad/Reshape_1
Я
@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity/gradients_1/mean_squared_error/Mul_grad/Reshape9^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*B
_class8
64loc:@gradients_1/mean_squared_error/Mul_grad/Reshape*
T0*
_output_shapes
:
£
Bgradients_1/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity1gradients_1/mean_squared_error/Mul_grad/Reshape_19^gradients_1/mean_squared_error/Mul_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/mean_squared_error/Mul_grad/Reshape_1*
T0*
_output_shapes
: 
К
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
ћ
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1Shape:mean_squared_error/num_present/broadcast_weights/ones_like*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
Ѕ
Wgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ShapeIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
б
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulMul4gradients_1/mean_squared_error/num_present_grad/Tile:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
ђ
Egradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumSumEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mulWgradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
У
Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeReshapeEgradients_1/mean_squared_error/num_present/broadcast_weights_grad/SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
ќ
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Mul%mean_squared_error/num_present/Select4gradients_1/mean_squared_error/num_present_grad/Tile*
T0*
_output_shapes
:
≤
Ggradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1SumGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/mul_1Ygradients_1/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ы
Kgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1ReshapeGgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Sum_1Igradients_1/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
ф
Rgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_depsNoOpJ^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeL^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1
Е
Zgradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityIgradients_1/mean_squared_error/num_present/broadcast_weights_grad/ReshapeS^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape*
T0*
_output_shapes
: 
Н
\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityKgradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1S^gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1*
T0*
_output_shapes
:
„
Pgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankRank\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Щ
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Щ
Wgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
П
Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/rangeRangeWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/startPgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/RankWgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
≈
Ogradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/SumSum\gradients_1/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1Qgradients_1/mean_squared_error/num_present/broadcast_weights/ones_like_grad/range*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Е
;gradients_1/mean_squared_error/SquaredDifference_grad/ShapeShape
baseline_1*
out_type0*
T0*
_output_shapes
:
Н
=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1ShapeSqueeze*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
Э
Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/mean_squared_error/SquaredDifference_grad/Shape=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ƒ
<gradients_1/mean_squared_error/SquaredDifference_grad/scalarConstA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
dtype0*
valueB
 *   @*
_output_shapes
: 
г
9gradients_1/mean_squared_error/SquaredDifference_grad/mulMul<gradients_1/mean_squared_error/SquaredDifference_grad/scalar@gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
ї
9gradients_1/mean_squared_error/SquaredDifference_grad/subSub
baseline_1SqueezeA^gradients_1/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
џ
;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mul9gradients_1/mean_squared_error/SquaredDifference_grad/mul9gradients_1/mean_squared_error/SquaredDifference_grad/sub*
T0*
_output_shapes
:
К
9gradients_1/mean_squared_error/SquaredDifference_grad/SumSum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Kgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ь
=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeReshape9gradients_1/mean_squared_error/SquaredDifference_grad/Sum;gradients_1/mean_squared_error/SquaredDifference_grad/Shape*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
О
;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1Sum;gradients_1/mean_squared_error/SquaredDifference_grad/mul_1Mgradients_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ч
?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape;gradients_1/mean_squared_error/SquaredDifference_grad/Sum_1=gradients_1/mean_squared_error/SquaredDifference_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
§
9gradients_1/mean_squared_error/SquaredDifference_grad/NegNeg?gradients_1/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes
:
 
Fgradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp>^gradients_1/mean_squared_error/SquaredDifference_grad/Reshape:^gradients_1/mean_squared_error/SquaredDifference_grad/Neg
в
Ngradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity=gradients_1/mean_squared_error/SquaredDifference_grad/ReshapeG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/mean_squared_error/SquaredDifference_grad/Reshape*
T0*#
_output_shapes
:€€€€€€€€€
—
Pgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity9gradients_1/mean_squared_error/SquaredDifference_grad/NegG^gradients_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/mean_squared_error/SquaredDifference_grad/Neg*
T0*
_output_shapes
:
А
gradients_1/Squeeze_grad/ShapeShape"baseline/fully_connected_1/BiasAdd*
out_type0*
T0*
_output_shapes
:
Ё
 gradients_1/Squeeze_grad/ReshapeReshapePgradients_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1gradients_1/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
ђ
?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad gradients_1/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:
±
Dgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp!^gradients_1/Squeeze_grad/Reshape@^gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad
®
Lgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity gradients_1/Squeeze_grad/ReshapeE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Squeeze_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€
џ
Ngradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGradE^gradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/baseline/fully_connected_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
Т
9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulMatMulLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency'baseline/fully_connected_1/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
Б
;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1MatMulbaseline/fully_connected/ReluLgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
≈
Cgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_depsNoOp:^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul<^gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1
Ў
Kgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity9gradients_1/baseline/fully_connected_1/MatMul_grad/MatMulD^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
’
Mgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity;gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1D^gradients_1/baseline/fully_connected_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/baseline/fully_connected_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
б
7gradients_1/baseline/fully_connected/Relu_grad/ReluGradReluGradKgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependencybaseline/fully_connected/Relu*
T0*'
_output_shapes
:€€€€€€€€€
Ѕ
=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
ƒ
Bgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/Relu_grad/ReluGrad>^gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad
“
Jgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/Relu_grad/ReluGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/Relu_grad/ReluGrad*
T0*'
_output_shapes
:€€€€€€€€€
”
Lgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGradC^gradients_1/baseline/fully_connected/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/baseline/fully_connected/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
М
7gradients_1/baseline/fully_connected/MatMul_grad/MatMulMatMulJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency%baseline/fully_connected/weights/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€
б
9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1MatMuloJgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
њ
Agradients_1/baseline/fully_connected/MatMul_grad/tuple/group_depsNoOp8^gradients_1/baseline/fully_connected/MatMul_grad/MatMul:^gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1
–
Igradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependencyIdentity7gradients_1/baseline/fully_connected/MatMul_grad/MatMulB^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€
Ќ
Kgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1Identity9gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1B^gradients_1/baseline/fully_connected/MatMul_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/baseline/fully_connected/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
Ф
beta1_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *fff?*
_output_shapes
: 
•
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
»
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
В
beta1_power_1/readIdentitybeta1_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
Ф
beta2_power_1/initial_valueConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB
 *wЊ?*
_output_shapes
: 
•
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
»
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
: 
В
beta2_power_1/readIdentitybeta2_power_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
Ѕ
7baseline/fully_connected/weights/Adam/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB*    *
_output_shapes

:
ќ
%baseline/fully_connected/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
Э
,baseline/fully_connected/weights/Adam/AssignAssign%baseline/fully_connected/weights/Adam7baseline/fully_connected/weights/Adam/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
ї
*baseline/fully_connected/weights/Adam/readIdentity%baseline/fully_connected/weights/Adam*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
√
9baseline/fully_connected/weights/Adam_1/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@baseline/fully_connected/weights*
valueB*    *
_output_shapes

:
–
'baseline/fully_connected/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*3
_class)
'%loc:@baseline/fully_connected/weights*
shared_name 
£
.baseline/fully_connected/weights/Adam_1/AssignAssign'baseline/fully_connected/weights/Adam_19baseline/fully_connected/weights/Adam_1/Initializer/zeros*
validate_shape(*3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking(*
T0*
_output_shapes

:
њ
,baseline/fully_connected/weights/Adam_1/readIdentity'baseline/fully_connected/weights/Adam_1*3
_class)
'%loc:@baseline/fully_connected/weights*
T0*
_output_shapes

:
Ј
6baseline/fully_connected/biases/Adam/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB*    *
_output_shapes
:
ƒ
$baseline/fully_connected/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
Х
+baseline/fully_connected/biases/Adam/AssignAssign$baseline/fully_connected/biases/Adam6baseline/fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
і
)baseline/fully_connected/biases/Adam/readIdentity$baseline/fully_connected/biases/Adam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
:
є
8baseline/fully_connected/biases/Adam_1/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@baseline/fully_connected/biases*
valueB*    *
_output_shapes
:
∆
&baseline/fully_connected/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@baseline/fully_connected/biases*
shared_name 
Ы
-baseline/fully_connected/biases/Adam_1/AssignAssign&baseline/fully_connected/biases/Adam_18baseline/fully_connected/biases/Adam_1/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking(*
T0*
_output_shapes
:
Є
+baseline/fully_connected/biases/Adam_1/readIdentity&baseline/fully_connected/biases/Adam_1*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
:
≈
9baseline/fully_connected_1/weights/Adam/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB*    *
_output_shapes

:
“
'baseline/fully_connected_1/weights/Adam
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
•
.baseline/fully_connected_1/weights/Adam/AssignAssign'baseline/fully_connected_1/weights/Adam9baseline/fully_connected_1/weights/Adam/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
Ѕ
,baseline/fully_connected_1/weights/Adam/readIdentity'baseline/fully_connected_1/weights/Adam*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
«
;baseline/fully_connected_1/weights/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@baseline/fully_connected_1/weights*
valueB*    *
_output_shapes

:
‘
)baseline/fully_connected_1/weights/Adam_1
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*5
_class+
)'loc:@baseline/fully_connected_1/weights*
shared_name 
Ђ
0baseline/fully_connected_1/weights/Adam_1/AssignAssign)baseline/fully_connected_1/weights/Adam_1;baseline/fully_connected_1/weights/Adam_1/Initializer/zeros*
validate_shape(*5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking(*
T0*
_output_shapes

:
≈
.baseline/fully_connected_1/weights/Adam_1/readIdentity)baseline/fully_connected_1/weights/Adam_1*5
_class+
)'loc:@baseline/fully_connected_1/weights*
T0*
_output_shapes

:
ї
8baseline/fully_connected_1/biases/Adam/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB*    *
_output_shapes
:
»
&baseline/fully_connected_1/biases/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
Э
-baseline/fully_connected_1/biases/Adam/AssignAssign&baseline/fully_connected_1/biases/Adam8baseline/fully_connected_1/biases/Adam/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
Ї
+baseline/fully_connected_1/biases/Adam/readIdentity&baseline/fully_connected_1/biases/Adam*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
:
љ
:baseline/fully_connected_1/biases/Adam_1/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@baseline/fully_connected_1/biases*
valueB*    *
_output_shapes
:
 
(baseline/fully_connected_1/biases/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*4
_class*
(&loc:@baseline/fully_connected_1/biases*
shared_name 
£
/baseline/fully_connected_1/biases/Adam_1/AssignAssign(baseline/fully_connected_1/biases/Adam_1:baseline/fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking(*
T0*
_output_shapes
:
Њ
-baseline/fully_connected_1/biases/Adam_1/readIdentity(baseline/fully_connected_1/biases/Adam_1*4
_class*
(&loc:@baseline/fully_connected_1/biases*
T0*
_output_shapes
:
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *П¬х<*
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
 *wЊ?*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
у
8Adam_1/update_baseline/fully_connected/weights/ApplyAdam	ApplyAdam baseline/fully_connected/weights%baseline/fully_connected/weights/Adam'baseline/fully_connected/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonKgradients_1/baseline/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *3
_class)
'%loc:@baseline/fully_connected/weights*
use_locking( *
T0*
_output_shapes

:
л
7Adam_1/update_baseline/fully_connected/biases/ApplyAdam	ApplyAdambaseline/fully_connected/biases$baseline/fully_connected/biases/Adam&baseline/fully_connected/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonLgradients_1/baseline/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
:
€
:Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam	ApplyAdam"baseline/fully_connected_1/weights'baseline/fully_connected_1/weights/Adam)baseline/fully_connected_1/weights/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonMgradients_1/baseline/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *5
_class+
)'loc:@baseline/fully_connected_1/weights*
use_locking( *
T0*
_output_shapes

:
ч
9Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam	ApplyAdam!baseline/fully_connected_1/biases&baseline/fully_connected_1/biases/Adam(baseline/fully_connected_1/biases/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/baseline/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *4
_class*
(&loc:@baseline/fully_connected_1/biases*
use_locking( *
T0*
_output_shapes
:
ц

Adam_1/mulMulbeta1_power_1/readAdam_1/beta19^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
∞
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
ш
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta29^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam*2
_class(
&$loc:@baseline/fully_connected/biases*
T0*
_output_shapes
: 
і
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
validate_shape(*2
_class(
&$loc:@baseline/fully_connected/biases*
use_locking( *
T0*
_output_shapes
: 
Ю
Adam_1NoOp9^Adam_1/update_baseline/fully_connected/weights/ApplyAdam8^Adam_1/update_baseline/fully_connected/biases/ApplyAdam;^Adam_1/update_baseline/fully_connected_1/weights/ApplyAdam:^Adam_1/update_baseline/fully_connected_1/biases/ApplyAdam^Adam_1/Assign^Adam_1/Assign_1
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
Adam_1"ъ
trainable_variablesвя
Ќ
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
ј
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
’
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
»
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
Q
	log_std:0log_std/Assignlog_std/read:02$log_std/Initializer/random_uniform:0
µ
"baseline/fully_connected/weights:0'baseline/fully_connected/weights/Assign'baseline/fully_connected/weights/read:02=baseline/fully_connected/weights/Initializer/random_uniform:0
®
!baseline/fully_connected/biases:0&baseline/fully_connected/biases/Assign&baseline/fully_connected/biases/read:023baseline/fully_connected/biases/Initializer/zeros:0
љ
$baseline/fully_connected_1/weights:0)baseline/fully_connected_1/weights/Assign)baseline/fully_connected_1/weights/read:02?baseline/fully_connected_1/weights/Initializer/random_uniform:0
∞
#baseline/fully_connected_1/biases:0(baseline/fully_connected_1/biases/Assign(baseline/fully_connected_1/biases/read:025baseline/fully_connected_1/biases/Initializer/zeros:0"м+
	variablesё+џ+
Ќ
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
ј
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
’
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
»
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
Q
	log_std:0log_std/Assignlog_std/read:02$log_std/Initializer/random_uniform:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
Ў
-policy_network/fully_connected/weights/Adam:02policy_network/fully_connected/weights/Adam/Assign2policy_network/fully_connected/weights/Adam/read:02?policy_network/fully_connected/weights/Adam/Initializer/zeros:0
а
/policy_network/fully_connected/weights/Adam_1:04policy_network/fully_connected/weights/Adam_1/Assign4policy_network/fully_connected/weights/Adam_1/read:02Apolicy_network/fully_connected/weights/Adam_1/Initializer/zeros:0
‘
,policy_network/fully_connected/biases/Adam:01policy_network/fully_connected/biases/Adam/Assign1policy_network/fully_connected/biases/Adam/read:02>policy_network/fully_connected/biases/Adam/Initializer/zeros:0
№
.policy_network/fully_connected/biases/Adam_1:03policy_network/fully_connected/biases/Adam_1/Assign3policy_network/fully_connected/biases/Adam_1/read:02@policy_network/fully_connected/biases/Adam_1/Initializer/zeros:0
а
/policy_network/fully_connected_1/weights/Adam:04policy_network/fully_connected_1/weights/Adam/Assign4policy_network/fully_connected_1/weights/Adam/read:02Apolicy_network/fully_connected_1/weights/Adam/Initializer/zeros:0
и
1policy_network/fully_connected_1/weights/Adam_1:06policy_network/fully_connected_1/weights/Adam_1/Assign6policy_network/fully_connected_1/weights/Adam_1/read:02Cpolicy_network/fully_connected_1/weights/Adam_1/Initializer/zeros:0
№
.policy_network/fully_connected_1/biases/Adam:03policy_network/fully_connected_1/biases/Adam/Assign3policy_network/fully_connected_1/biases/Adam/read:02@policy_network/fully_connected_1/biases/Adam/Initializer/zeros:0
д
0policy_network/fully_connected_1/biases/Adam_1:05policy_network/fully_connected_1/biases/Adam_1/Assign5policy_network/fully_connected_1/biases/Adam_1/read:02Bpolicy_network/fully_connected_1/biases/Adam_1/Initializer/zeros:0
\
log_std/Adam:0log_std/Adam/Assignlog_std/Adam/read:02 log_std/Adam/Initializer/zeros:0
d
log_std/Adam_1:0log_std/Adam_1/Assignlog_std/Adam_1/read:02"log_std/Adam_1/Initializer/zeros:0
µ
"baseline/fully_connected/weights:0'baseline/fully_connected/weights/Assign'baseline/fully_connected/weights/read:02=baseline/fully_connected/weights/Initializer/random_uniform:0
®
!baseline/fully_connected/biases:0&baseline/fully_connected/biases/Assign&baseline/fully_connected/biases/read:023baseline/fully_connected/biases/Initializer/zeros:0
љ
$baseline/fully_connected_1/weights:0)baseline/fully_connected_1/weights/Assign)baseline/fully_connected_1/weights/read:02?baseline/fully_connected_1/weights/Initializer/random_uniform:0
∞
#baseline/fully_connected_1/biases:0(baseline/fully_connected_1/biases/Assign(baseline/fully_connected_1/biases/read:025baseline/fully_connected_1/biases/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
ј
'baseline/fully_connected/weights/Adam:0,baseline/fully_connected/weights/Adam/Assign,baseline/fully_connected/weights/Adam/read:029baseline/fully_connected/weights/Adam/Initializer/zeros:0
»
)baseline/fully_connected/weights/Adam_1:0.baseline/fully_connected/weights/Adam_1/Assign.baseline/fully_connected/weights/Adam_1/read:02;baseline/fully_connected/weights/Adam_1/Initializer/zeros:0
Љ
&baseline/fully_connected/biases/Adam:0+baseline/fully_connected/biases/Adam/Assign+baseline/fully_connected/biases/Adam/read:028baseline/fully_connected/biases/Adam/Initializer/zeros:0
ƒ
(baseline/fully_connected/biases/Adam_1:0-baseline/fully_connected/biases/Adam_1/Assign-baseline/fully_connected/biases/Adam_1/read:02:baseline/fully_connected/biases/Adam_1/Initializer/zeros:0
»
)baseline/fully_connected_1/weights/Adam:0.baseline/fully_connected_1/weights/Adam/Assign.baseline/fully_connected_1/weights/Adam/read:02;baseline/fully_connected_1/weights/Adam/Initializer/zeros:0
–
+baseline/fully_connected_1/weights/Adam_1:00baseline/fully_connected_1/weights/Adam_1/Assign0baseline/fully_connected_1/weights/Adam_1/read:02=baseline/fully_connected_1/weights/Adam_1/Initializer/zeros:0
ƒ
(baseline/fully_connected_1/biases/Adam:0-baseline/fully_connected_1/biases/Adam/Assign-baseline/fully_connected_1/biases/Adam/read:02:baseline/fully_connected_1/biases/Adam/Initializer/zeros:0
ћ
*baseline/fully_connected_1/biases/Adam_1:0/baseline/fully_connected_1/biases/Adam_1/Assign/baseline/fully_connected_1/biases/Adam_1/read:02<baseline/fully_connected_1/biases/Adam_1/Initializer/zeros:0"(
losses

mean_squared_error/value:0"н±
cond_contextџ±„±
—
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textZmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *Ј
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
7mean_squared_error/assert_broadcastable/weights/shape:0±
6mean_squared_error/assert_broadcastable/values/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1і
7mean_squared_error/assert_broadcastable/weights/shape:0ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
ѓ
\mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Х
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0Њ
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Ж
@mean_squared_error/assert_broadcastable/is_valid_shape/cond_text@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0 *Ї
3mean_squared_error/assert_broadcastable/is_scalar:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0x
3mean_squared_error/assert_broadcastable/is_scalar:0Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
ƒ
Bmean_squared_error/assert_broadcastable/is_valid_shape/cond_text_1@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0*ш
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
7mean_squared_error/assert_broadcastable/weights/shape:0Я
5mean_squared_error/assert_broadcastable/values/rank:0fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0ѓ
6mean_squared_error/assert_broadcastable/values/shape:0umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0≤
7mean_squared_error/assert_broadcastable/weights/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Ґ
6mean_squared_error/assert_broadcastable/weights/rank:0hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
М
=mean_squared_error/assert_broadcastable/AssertGuard/cond_text=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0 *…
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0
ґ
?mean_squared_error/assert_broadcastable/AssertGuard/cond_text_1=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0*у
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
7mean_squared_error/assert_broadcastable/weights/shape:0Е
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0|
3mean_squared_error/assert_broadcastable/is_scalar:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
6mean_squared_error/assert_broadcastable/values/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0А
7mean_squared_error/assert_broadcastable/weights/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0
к
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textxmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *ц
Ъmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Ъmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Ъmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Рmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Мmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Чmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Тmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Оmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Нmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Иmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Тmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Сmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Сmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Лmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Гmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Бmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0о
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1с
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0Чmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
џ	
zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*з
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0ъ
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
≥
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *Н
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0і
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
п
`mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*…
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Бmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Дmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
Жmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0№
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0Дmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0м
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0Уmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0п
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0Хmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0я
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0Жmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
ј
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *£
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0
А
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*г
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
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0Є
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0Ѕ
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0Љ
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0ї
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0"H
	summaries;
9
Avg_Reward:0
Max_Reward:0
Std_Reward:0
Eval_Reward:0"£
model_variablesПМ
Ќ
(policy_network/fully_connected/weights:0-policy_network/fully_connected/weights/Assign-policy_network/fully_connected/weights/read:02Cpolicy_network/fully_connected/weights/Initializer/random_uniform:0
ј
'policy_network/fully_connected/biases:0,policy_network/fully_connected/biases/Assign,policy_network/fully_connected/biases/read:029policy_network/fully_connected/biases/Initializer/zeros:0
’
*policy_network/fully_connected_1/weights:0/policy_network/fully_connected_1/weights/Assign/policy_network/fully_connected_1/weights/read:02Epolicy_network/fully_connected_1/weights/Initializer/random_uniform:0
»
)policy_network/fully_connected_1/biases:0.policy_network/fully_connected_1/biases/Assign.policy_network/fully_connected_1/biases/read:02;policy_network/fully_connected_1/biases/Initializer/zeros:0
µ
"baseline/fully_connected/weights:0'baseline/fully_connected/weights/Assign'baseline/fully_connected/weights/read:02=baseline/fully_connected/weights/Initializer/random_uniform:0
®
!baseline/fully_connected/biases:0&baseline/fully_connected/biases/Assign&baseline/fully_connected/biases/read:023baseline/fully_connected/biases/Initializer/zeros:0
љ
$baseline/fully_connected_1/weights:0)baseline/fully_connected_1/weights/Assign)baseline/fully_connected_1/weights/read:02?baseline/fully_connected_1/weights/Initializer/random_uniform:0
∞
#baseline/fully_connected_1/biases:0(baseline/fully_connected_1/biases/Assign(baseline/fully_connected_1/biases/read:025baseline/fully_connected_1/biases/Initializer/zeros:0 	МЁX       ҐЛ/М	чд@§÷A*M


Avg_Reward'дЕ@


Max_Reward  АA


Std_RewardТ*>

Eval_Reward  @@®N\ФZ       oіњ	[=д@§÷A*M


Avg_Reward9∞Ф@


Max_Reward  –A


Std_Rewardu[>

Eval_Reward   @ћ	µ£Z       oіњ	‘р\д@§÷A*M


Avg_RewardҐЉ¶@


Max_Reward  ИA


Std_Reward{ o>

Eval_Reward  @@.ШМZ       oіњ	Cбzд@§÷A*M


Avg_Rewardж(љ@


Max_Reward  јA


Std_RewardпhЪ>

Eval_Reward  А@АЅZ       oіњ	ҐaЩд@§÷A*M


Avg_Reward  Џ@


Max_Reward  –A


Std_Reward<%љ>

Eval_Reward  †@≥ўчАZ       oіњ	¶+Јд@§÷A*M


Avg_RewardH®ё@


Max_Reward  ∞A


Std_Reward≈р≥>

Eval_Reward  ј@ы≈y'Z       oіњ	gЮ”д@§÷A*M


Avg_Rewardо|€@


Max_Reward  аA


Std_Reward§Eо>

Eval_Reward  AjµЩZ       oіњ	<Япд@§÷A*M


Avg_Reward јч@


Max_Reward  B


Std_RewardНnЅ>

Eval_Reward  А@'ћL$Z       oіњ	сфе@§÷A*M


Avg_RewardД:A


Max_Reward   B


Std_RewardYG?

Eval_Reward  АAЩmo’Z       oіњ	„Џ'е@§÷A	*M


Avg_RewardџC>A


Max_Reward  B


Std_RewardZ{>?

Eval_Reward  PAе°Ч…Z       oіњ	їBе@§÷A
*M


Avg_Reward^CQA


Max_Reward  (B


Std_RewardОE[?

Eval_Reward  @@{o—≈Z       oіњ	™\е@§÷A*M


Avg_Rewardњп{A


Max_Reward  ШB


Std_Reward°Ї’?

Eval_Reward  –AЭni…Z       oіњ	ЦЌvе@§÷A*M


Avg_RewardєІЙA


Max_Reward  B


Std_RewardЕ≈Ч?

Eval_Reward   AvdZ       oіњ	CТСе@§÷A*M


Avg_Reward–!ХA


Max_Reward  ВB


Std_Reward№iЌ?

Eval_Reward  а@1?ќZ       oіњ	х™е@§÷A*M


Avg_RewardJЯђA


Max_Reward  HB


Std_Reward±nв?

Eval_Reward  »AъЗ¶sZ       oіњ	…™√е@§÷A*M


Avg_RewardIТЛA


Max_Reward  DB


Std_Rewardyє?

Eval_Reward  †@™їCZZ       oіњ	’Ёе@§÷A*M


Avg_RewardeMєA


Max_Reward  ґB


Std_Rewardc@

Eval_Reward  0A”rЂэZ       oіњ	tЦхе@§÷A*M


Avg_RewardСЕђA


Max_Reward  8B


Std_Rewardѓg—?

Eval_Reward  BџЎсZ       oіњ	oж@§÷A*M


Avg_RewardЬёђA


Max_Reward  ,B


Std_RewardњЬѓ?

Eval_Reward  @AЂ`pОZ       oіњ	”*ж@§÷A*M


Avg_Reward<вґA


Max_Reward  `B


Std_Rewardђўх?

Eval_Reward  ИA¬zƒZ       oіњ	ЄsDж@§÷A*M


Avg_Reward0и≥A


Max_Reward  tB


Std_RewardґЊе?

Eval_Reward  »AКДѓЎZ       oіњ	}≤_ж@§÷A*M


Avg_Rewardф<їA


Max_Reward  АB


Std_Reward©kф?

Eval_Reward  АA≠ј“MZ       oіњ	Нс{ж@§÷A*M


Avg_Reward  ОA


Max_Reward  (B


Std_RewardЧ}?

Eval_Reward  pAїџєZ       oіњ	<ДШж@§÷A*M


Avg_Rewardњ≥©A


Max_Reward  8B


Std_Rewardї…∞?

Eval_Reward  рAЁt/rZ       oіњ	ў≥ж@§÷A*M


Avg_Reward}єA


Max_Reward  lB


Std_Reward°сў?

Eval_Reward  †ArТ~XZ       oіњ	TОЌж@§÷A*M


Avg_Reward  РA


Max_Reward  8B


Std_Reward6О?

Eval_Reward  A]ђ<GZ       oіњ		Ййж@§÷A*M


Avg_Reward|рqA


Max_Reward  B


Std_RewardтD?

Eval_Reward  PAF ЦюZ       oіњ	 ≥з@§÷A*M


Avg_RewardU’£A


Max_Reward  4B


Std_Rewardo∞Р?

Eval_Reward  »AD$Z       oіњ	ЧОз@§÷A*M


Avg_RewardЗтќA


Max_Reward  <B


Std_Reward#Ж∞?

Eval_Reward  B
ьZ       oіњ	‘Ш9з@§÷A*M


Avg_Reward√ЉA


Max_Reward  dB


Std_Reward;шї?

Eval_Reward  –Aг^ЦZ       oіњ	iбRз@§÷A*M


Avg_RewardЊеЋA


Max_Reward  \B


Std_RewardDg“?

Eval_Reward  ЎAђaЃНZ       oіњ	dlз@§÷A*M


Avg_Rewardy—A


Max_Reward  TB


Std_Rewardlr¬?

Eval_Reward  †A—бqZ       oіњ	Т Ез@§÷A *M


Avg_Reward‘бA


Max_Reward  ЖB


Std_Rewardђъ?

Eval_Reward  ®A∆л*ЊZ       oіњ	 ≤†з@§÷A!*M


Avg_RewardсяA


Max_Reward  \B


Std_Reward\“ж?

Eval_Reward  0A'@T¬Z       oіњ	6еєз@§÷A"*M


Avg_Rewardґ÷ъA


Max_Reward  HB


Std_Reward0Iл?

Eval_Reward  јAq[jЖZ       oіњ	І”з@§÷A#*M


Avg_Reward]мA


Max_Reward  @B


Std_Reward√'љ?

Eval_Reward  B≈∞зЯZ       oіњ	3Ёлз@§÷A$*M


Avg_RewardпоB


Max_Reward  КB


Std_RewardпЌ@

Eval_Reward  `BР'£zZ       oіњ	*и@§÷A%*M


Avg_Reward АйA


Max_Reward  PB


Std_Rewardбюг?

Eval_Reward  BGf{Z       oіњ	#hи@§÷A&*M


Avg_Reward @уA


Max_Reward  XB


Std_Reward

@

Eval_Reward  –A¶^zвZ       oіњ	Ug:и@§÷A'*M


Avg_Reward=
B


Max_Reward  ™B


Std_RewardйB@

Eval_Reward  »A=С–™Z       oіњ	1мTи@§÷A(*M


Avg_RewardRЄB


Max_Reward  МB


Std_RewardyЈF@

Eval_Reward  ЎA-®п'Z       oіњ	ж@nи@§÷A)*M


Avg_Rewardz”'B


Max_Reward  ђB


Std_RewardrT@

Eval_Reward  hBУjSZ       oіњ	"pЗи@§÷A**M


Avg_RewardyQB


Max_Reward  C


Std_RewardcT…@

Eval_Reward  dB°hґZ       oіњ	R*Ґи@§÷A+*M


Avg_RewardU’B


Max_Reward  ™B


Std_Rewardс-;@

Eval_Reward  \BFнr)Z       oіњ	“4Љи@§÷A,*M


Avg_Reward
„B


Max_Reward  ЃB


Std_RewardЅVl@

Eval_Reward  0BvжйZ       oіњ	9с÷и@§÷A-*M


Avg_RewardЌћB


Max_Reward  ВB


Std_Reward{®*@

Eval_Reward  ∞A\}`€Z       oіњ	ЇВси@§÷A.*M


Avg_Rewardе5PB


Max_Reward  дB


Std_RewardІѓ@

Eval_Reward  8BE∞х§Z       oіњ	°Gй@§÷A/*M


Avg_Reward  !B


Max_Reward  ЦB


Std_Rewardф^7@

Eval_Reward  BЋ=2 Z       oіњ	я3й@§÷A0*M


Avg_Rewardnџ2B


Max_Reward  ЮB


Std_RewardМEt@

Eval_Reward  Bћ∆Z       oіњ	РOй@§÷A1*M


Avg_RewardЪЩAB


Max_Reward  јB


Std_RewardCжI@

Eval_Reward  <BНX^Z       oіњ	_‘jй@§÷A2*M


Avg_RewardffvB


Max_Reward  иB


Std_RewardC∆√@

Eval_Reward  0B”XV;Z       oіњ	зsДй@§÷A3*M


Avg_Reward  uB


Max_Reward  ÷B


Std_Reward6LЅ@

Eval_Reward  HBМn?ЅZ       oіњ	&аЯй@§÷A4*M


Avg_Reward;СB


Max_Reward  C


Std_RewardkA

Eval_Reward  ДBб(8dZ       oіњ	цмїй@§÷A5*M


Avg_RewardџґБB


Max_Reward  #C


Std_Reward2CA

Eval_Reward  #CґIЊZ       oіњ	ѓ’й@§÷A6*M


Avg_Reward/ЇЃB


Max_Reward  ,C


Std_Rewardа+A

Eval_Reward  тBгФ•еZ       oіњ	Їjпй@§÷A7*M


Avg_RewardЂ™§B


Max_Reward  C


Std_RewardX4(A

Eval_Reward  ЉBzda≈Z       oіњ	bЗк@§÷A8*M


Avg_RewardџґsB


Max_Reward  жB


Std_RewardљЉз@

Eval_Reward  @Bс©јZ       oіњ	•№"к@§÷A9*M


Avg_Reward«qґB


Max_Reward  /C


Std_RewardЂgA

Eval_Reward  ОBg„ъRZ       oіњ	-
<к@§÷A:*M


Avg_Reward  C


Max_Reward АЃC


Std_Rewardд1B

Eval_Reward  ƒBќџ)Z       oіњ	ЈDUк@§÷A;*M


Avg_RewardU’C


Max_Reward  ЙC


Std_RewardNWB

Eval_Reward  9C]KтZ       oіњ	KЂnк@§÷A<*M


Avg_Reward  —B


Max_Reward АДC


Std_Reward°с√A

Eval_Reward АДC√(≠чZ       oіњ	УюЗк@§÷A=*M


Avg_RewardЂ™ЪC


Max_Reward  тC


Std_RewardзЃЅB

Eval_Reward АіCs= LZ       oіњ	N+£к@§÷A>*M


Avg_Reward  gC


Max_Reward  —C


Std_RewardАџ\B

Eval_Reward  8Cур=ЂZ       oіњ	uдїк@§÷A?*M


Avg_Reward @SC


Max_Reward јD


Std_RewardEдB

Eval_Reward  dB©PЬЛZ       oіњ	EД’к@§÷A@*M


Avg_Reward  iC


Max_Reward АЧC


Std_RewardjэEB

Eval_Reward  #Cг≈fNZ       oіњ	4Юнк@§÷AA*M


Avg_RewardU’БC


Max_Reward  ©C


Std_Reward÷3fB

Eval_Reward  °C÷¬<АZ       oіњ	&Пл@§÷AB*M


Avg_RewardUU[C


Max_Reward  ЪC


Std_RewardГ_B

Eval_Reward  ЪC$ЈyZ       oіњ	Л#л@§÷AC*M


Avg_Reward @ЉC


Max_Reward @D


Std_RewardtЪC

Eval_Reward @DЅ∆ќZ       oіњ	нЭ=л@§÷AD*M


Avg_Reward @&D


Max_Reward @&D


Std_Reward    

Eval_Reward @&D…>§Z       oіњ	Wл@§÷AE*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD.HґZ       oіњ	+џoл@§÷AF*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDЪн"±Z       oіњ	ZДЙл@§÷AG*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD«t"ПZ       oіњ	”Ґл@§÷AH*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD:ю\µZ       oіњ	DЉл@§÷AI*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD®+љфZ       oіњ	Я’л@§÷AJ*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDW ≈Z       oіњ	Яол@§÷AK*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDњ0еZ       oіњ	∞м@§÷AL*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDf(ґ.Z       oіњ	E, м@§÷AM*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD)C™Z       oіњ	5~9м@§÷AN*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD|PEЖZ       oіњ	яPRм@§÷AO*M


Avg_RewardЂ™РC


Max_Reward А$D


Std_Reward–C

Eval_Reward  CВƒЄ=Z       oіњ	D0kм@§÷AP*M


Avg_Rewardff:C


Max_Reward А”C


Std_Rewardх£ZB

Eval_Reward  1CO™хZ       oіњ	ХДм@§÷AQ*M


Avg_RewardIТюB


Max_Reward  ]C


Std_RewardіЫA

Eval_Reward  C'\ЪuZ       oіњ	≤Эм@§÷AR*M


Avg_RewardЌћGC


Max_Reward @0D


Std_Reward£™вB

Eval_Reward  аBuDцщZ       oіњ	#Ґґм@§÷AS*M


Avg_Reward  мC


Max_Reward  мC


Std_Reward    

Eval_Reward  мC1{jZ       oіњ	Hбќм@§÷AT*M


Avg_Reward ј D


Max_Reward ј D


Std_Reward    

Eval_Reward ј DоЋїZ       oіњ	Ћчжм@§÷AU*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD№ўMЯZ       oіњ	ХЬ€м@§÷AV*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDпS:)Z       oіњ	∆kн@§÷AW*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD€U *Z       oіњ	Й@2н@§÷AX*M


Avg_Reward АXD


Max_Reward АXD


Std_Reward    

Eval_Reward АXD_EіьZ       oіњ	Q/Mн@§÷AY*M


Avg_Reward @щC


Max_Reward @	D


Std_Rewardи’B

Eval_Reward @	DўјZ       oіњ	,Fgн@§÷AZ*M


Avg_Reward @іC


Max_Reward јD


Std_RewardpwлB

Eval_Reward јD/эіяZ       oіњ	Нѕн@§÷A[*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDрщР=Z       oіњ	ќ8Щн@§÷A\*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDы№нѕZ       oіњ	≥н@§÷A]*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDП«»QZ       oіњ	ЊҐћн@§÷A^*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDG§l—Z       oіњ	Єен@§÷A_*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zDВ…vцZ       oіњ	Ъ@€н@§÷A`*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD\VБZ       oіњ	Жђо@§÷Aa*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD$сТ)Z       oіњ	"q2о@§÷Ab*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD(ЗЃZ       oіњ	Ю^Lо@§÷Ac*M


Avg_Reward  zD


Max_Reward  zD


Std_Reward    

Eval_Reward  zD§НЮ