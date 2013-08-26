require 'torch'
require 'nn'
-- require 'kex'

IGNORE_CLASSIFICATION = false
UNSUPERVISED_CLASSIFICATION = false -- rather than using KL(p:q), where p is the observed distribution (one 1 and the rest 0's), use KL(q:q), which is equivalent to the negative entropy 
DEBUG_shrink = false -- don't require that the shrink value be non-negative, to facilitate the comparison of backpropagated gradients to forward-propagated parameter perturbations
DEBUG_L2 = false
DEBUG_L1 = false
DEBUG_OUTPUT = false
DEBUG_RF_TRAINING_SCALE = nil
PARAMETER_UPDATE_METHOD = 'optim' -- 'module' -- when updating using the optim package, each set of shared parameters is updated only once and gradients must be shared; when updating using the module methods of the nn package, every instance of a set of shared parameters is updated individually, and gradients should not be shared.  Make sure this matches setting in test_recpool_net.lua
FORCE_NONNEGATIVE_SHRINK_OUTPUT = true -- if the shrink output is non-negative, unrolled ISTA reconstructions tend to be poor unless there are more than twice as many hidden units as visible units, since about half of the hidden units will be prevented from growing smaller than zero, as would be required for optimal reconstruction
USE_FULL_SCALE_FOR_REPEATED_ISTA_MODULES = false -- if false, scale down the learning rate for the ISTA modules in proportion to the number of repeats
FULLY_NORMALIZE_ENC_FE_DICT = false -- if true, force the L2 norm of each row to be constant, rather than merely bounded the L2 norm above
FULLY_NORMALIZE_DEC_FE_DICT = true --false -- if true, force the L2 norm of each column to be constant; this is turned on below when using entropy or weighted-L1 regularizers
DEC_FE_DICT_NORM_VALUE = nil
NORMALIZE_ROWS_OF_ENC_FE_DICT = true
NORMALIZE_CLASS_DICT_OUTPUT = false
NORMALIZE_ROWS_OF_CLASS_DICT = true 
BOUND_ROWS_OF_CLASS_DICT = false -- USE WITH soft_aggregate class criterion
CLASS_DICT_BOUND = 5 --2 --5
CLASS_DICT_GRAD_SCALING = 0.2
LOCAL_CONNECTIONS_ONLY_EXPLAINING_AWAY_MATRIX = false -- filter the explaining_away weight matrix with a matrix that contains entries in the set {0,1} that only allows local connections between units
NO_BIAS_ON_EXPLAINING_AWAY = true -- bias on the explaining-away matrix is roughly equivalent to bias on the encoding_feature_extraction_dictionary
NORMALIZE_ROWS_OF_EXPLAINING_AWAY = false
BOUND_ELEMENTS_OF_EXPLAINING_AWAY = false -- EXPERIMENTAL CODE for use with UNSUPERVISED_CLASSIFICATION - keeps the diagonal elements of the explaining_away matrix from growing very large and alone inducing large activations for the categorical-units
EXPLAINING_AWAY_BOUND = 0.075
EXPLAINING_AWAY_BOUND_SCALE_FACTOR = 3 -- keep in mind that the expected row magnitude of the explaining away matrix increases linearly with the number of hidden units, and thus will tend to be larger than the rows of the encoding_feature_extraction_dictionary
ENC_CUMULATIVE_STEP_SIZE_INIT_1 = 0.2 -- for use with 2 ISTA iterations, to avoid instability in the initial dynamics
ENC_CUMULATIVE_STEP_SIZE_INIT_10 = 0.8 --1.25 -- USE 0.2 FOR STRONG TEMPLATE PRIMING -- CHANGE THIS BACK WHEN USING MORE THAN 2 ISTA ITERATIONS!!!
ENC_CUMULATIVE_STEP_SIZE_BOUND = 1.25 --1.25
NORMALIZE_ROWS_OF_P_FE_DICT = false
CREATE_BUFFER_ON_L1_LOSS = false --0.001
--MANUALLY_MAINTAIN_EXPLAINING_AWAY_DIAGONAL = true
CLASS_NLL_CRITERION_TYPE = nil --'soft_aggregate' --nil -- soft, soft_aggregate, hinge, nil -- use bound_rows_of_class_dict with soft_aggregate
USE_HETEROGENEOUS_L1_SCALING_FACTOR = false -- use a smaller L1 coefficient for the first few units than for the rest; my initial hope was that this would induce a small group of units with a reduced L1 coefficient to become categorical units, but instead of learning prototypes, they just learned a small basis set of traditional sparse parts, with many parts used to reconstruct each input
USE_L1_OVER_L2_NORM = false -- replace the L1 sparsifying norm on each layer with L1/L2; only the L1 norm need be subject to a scaling factor
USE_PROB_WEIGHTED_L1 = true -- replace the L1 sparsifying norm on each layer with L1/L2 weighted by softmax(L1/L2), plus the original L1; this is an approximation to the entropy-of-softmax regularizer
USE_HARD_ENTROPY = false -- replace the L1 sparsifying norm on each layer with the entropy of the L1-normalized hidden unit activation, pluse the original L1
LOW_PASS_FILTERED_ENTROPY = false --true
ELASTIC_NET_LOSS = 0.25 --true
--WEIGHTED_L1_SOFTMAX_SCALING = 0.35 -- FOR 2d SPIRAL ONLY!!!
--WEIGHTED_L1_PURE_L1_SCALING = 1.5 --1.0 --0.5 --1.5 --8 -- FOR 2d SPIRAL ONLY!!!

-- Because the hidden units are L2-normalized before the softmax, only a limited domain of the entropy loss is accessible.  It's essential that the entropy loss plateau within this domain.  With softmax_scaling = 0.875, the plateau begins when the activity of a single hidden unit reaches about 0.75 or 1.  With softmax_scaling = 0.875*2, the plateau begins closer to 0.25 or 0.5.  Since softmax scaling is applied after the L2 normalization, it increases the range of the entropy loss, necessitating that the entropy loss be scaled down so that the maximum gradient is kept roughly constant.  When the increase in softmax_scaling is properly balanced with the decrease in entropy_scaling, the hidden unit configurations at which the entropy gradient is maximal is shifted to configurations of intermediate entropy, whereas previously the maximal gradients occured at the lowest obtainable entropies.  As a result, the entropy loss will pull towards low entropy, but will gradually reduce its influence as the representation is dominated by a single large input.  Previously, the entropy gradient only became larger as the entropy decreased, leading to instability.
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 -- for MNIST
--WEIGHTED_L1_PURE_L1_SCALING = 1.5 --1 --1.5 --1.2 -- for MNIST
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 2 --0.9375 --0.875 -- for CIFAR -- USE THIS ONE

--WEIGHTED_L1_ENTROPY_SCALING = 0.2 -- general case

-- for 8x8 CIFAR
--local cifar_scaling = 0.5 -- 0.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 10 -- 12 is good without any entropy 
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 3.75 -- 4.5 is too large (just learns one categorical unit); 3 is too small (all units become quasi-categorical)
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 5.25 --4.5 -- 400 hidden units.  5.25 is too large.  Although it seems to perform well initially, eventually categorical units explode and are suppressed.  All units then become pseudo-categorical.  4.5 is better and continues to yield distinct populations of part- and categorical-units with extensive training, but part-units are still pulled to be pseudo-categorical.  Try 4.0.

-- for 12x12 CIFAR
--local cifar_scaling = 0.125 -- 0.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 --5 is too large with 1.5 entropy -- 4 is too small with 1.5 entropy -- 6 is the largest that produces sparse codes with reasonable reconstruction accuracy -- for CIFAR
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 1.5 -- 1.75 is too large with 4 entropy; 1.5 is too small with 4 entropy, but too large with 5 entropy -- CIFAR

-- for 12x12 CIFAR with 400 hidden units
--local cifar_scaling = 1 -- 0.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 -- 10 is too large, even without any entropy
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 2.5 * 0.0625 * 0.875 --2.0 -- 2.5 is too large.  Part-units are strongly pulled towards being pseudo-categorical, although a continuum remains even with continued training.  In contrast, initial performance is great, and exhibits a clear dichotomy between part- and categorical-units.

-- for 12x12 CIFAR with 400 hidden units, no L1 norm!
--local cifar_scaling = 0.5 --1 -- 0.5
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 2.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 -- 10 is too large, even without any entropy
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 0.15 --2.0 -- 2.5 is too large.  Part-units are strongly pulled towards being pseudo-categorical, although a continuum remains even with continued training.  In contrast, initial performance is great, and exhibits a clear dichotomy between part- and categorical-units.


-- for 12x12 Berkeley with 400 hidden units, no L2 norm!
--local cifar_scaling = 0.5 --0.1 --0.5 
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 1.25 --2.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 --5 -- 10 is too large, even without any entropy
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 1.5 --1 --1 --0.25 --0.15 -- softmax entropy


-- for 12x12 Berkeley with 400 hidden units, *with* L2 norm, adjusted to yield pooling-like behavior as in MNIST
local cifar_scaling = 0.5 --0.1 --0.5 
WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 1.25
WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 --5 -- 10 is too large, even without any entropy
WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 2 --2 --2 --1 --0.25 --0.15 -- softmax entropy

-- for 12x12 Berkeley with 400 hidden units, input divided by average L2 norm and then bounded at L2 norm = 1, *with* L2 norm in entropy calculation
-- (0.5, 0.875 * 2.0, 5, 0.2)
--local cifar_scaling = 0.5 
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 1.0 -- keep in mind that the largest inputs are smaller with normalization relative to their former values, so they are smaller relative to the appended constant
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 3 --2



-- for 12x12 Berkeley with 400 hidden units, hard entropy
--local cifar_scaling = 1 --0.1 --0.5 
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 2.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 5 --5 -- 10 is too large, even without any entropy
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 0.25 --0.001 --0.0025 -- 0.01 --0.025 --0.1 --0.25 --1 -- hard entropy, L1 normed


-- for 12x12 Berkeley with 400 hidden units, no L1 norm, unnormed inputs (L2); but any global scaling of the input is roughly equivalent to a rescaling of the loss function components, although not equally
--local cifar_scaling = 0.05 --0.5 
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 2.5
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 30 --5 --5 -- 10 is too large, even without any entropy
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 0.3 --0.15 


-- for 16x16 CIFAR
--local cifar_scaling = 0.5 --1 --2
--WEIGHTED_L1_PURE_L1_SCALING = cifar_scaling * 2 --5 is too large with 1.5 entropy -- 4 is too small with 1.5 entropy -- 6 is the largest that produces sparse codes with reasonable reconstruction accuracy -- for CIFAR
-- O.875 went to nans with scaling 1, but otherwise seemed to be moving towards categorical units
--WEIGHTED_L1_ENTROPY_SCALING = cifar_scaling * 0.875 --0.3 is too small, 0.5 is too small, 0.75 is too small; 1.0 is too large, 0.875 is too large (with scaling 1) 0.75 is too large (with scaling 2)

--WEIGHTED_L1_ENTROPY_SCALING = 0.3 -- 400 hidden units; when viewed as a weighted L1 loss, -\sum_i e^x_i / (\sum_j e^x_j) * log(e^x_i / (\sum_j e^x_j)) ~ -\sum_i e^x_i / (\sum_j e^x_j) * x_i, then since x is normalized to have L2 norm equal to 1, if we assume that only one unit is significantly active, then the entropy is e^1 / (k - 1 + e^1) * 1, and so is scaled down by a factor approximately equal to the number of hidden units.  When we double the number of hidden units, we should probably double the entropy scaling
L2_RECONSTRUCTION_SCALING_FACTOR = cifar_scaling * 0.25 * ((28*28) / (12*12)) -- CIFAR ; otherwise use 1


-- for MNIST with 400 hidden units, no L2 norm!
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 * 2.5 -- for MNIST
--WEIGHTED_L1_PURE_L1_SCALING = 1.5 --1 --1.5 --1.2 -- for MNIST
--WEIGHTED_L1_ENTROPY_SCALING = 0.02 -- 0.04 is too large, but is close
--L2_RECONSTRUCTION_SCALING_FACTOR = 1

-- for MNIST with 400 hidden units, *with* L2 norm
--WEIGHTED_L1_SOFTMAX_SCALING = 0.875 --0.875 -- for MNIST
--WEIGHTED_L1_PURE_L1_SCALING = 1.5 --1 --1.5 --1.2 -- for MNIST
--WEIGHTED_L1_ENTROPY_SCALING = 0.5 --0.2 -- general case
--L2_RECONSTRUCTION_SCALING_FACTOR = 1

--USE_L2_NORM_FOR_WEIGHTED_L1 = false --true
--APPEND_CONSTANT_VALUES = nil --{0.1}

USE_L2_NORM_FOR_WEIGHTED_L1 = true
APPEND_CONSTANT_VALUES = {0.1, 0.1}

--USE_L2_NORM_FOR_WEIGHTED_L1 = true
--APPEND_CONSTANT_VALUES = {0.1}




GROUP_SPARISTY_TEN_FIXED_GROUPS = false -- sets scaling of gradient for classification dictionary to 0, intializes it to consist of ten uniform disjoint groups, and replaces logistic regression with square root of sum of squares
if GROUP_SPARISTY_TEN_FIXED_GROUPS then
   CLASS_DICT_GRAD_SCALING = 0
end

if UNSUPERVISED_CLASSIFICATION then -- use an entropy-like regularizer, although it is applied in place of the usual logistic loss in an awkward manner
   --BOUND_ELEMENTS_OF_EXPLAINING_AWAY = true --otherwise, the diagonals of the categorical units grow very large, whereas the other recurrent inputs to the categorical units remain relatively small
   FULLY_NORMALIZE_DEC_FE_DICT = true -- otherwise, a single unit can always become very active to reduce the entropy of the hidden representation, and just represent the average input 
end

if USE_L1_OVER_L2_NORM or USE_PROB_WEIGHTED_L1 then -- these regularizers favor a single large value
   FULLY_NORMALIZE_DEC_FE_DICT = true -- otherwise, a single unit can always become very active to reduce the entropy of the hidden representation, and just represent the average input 
end

if NORMALIZE_ROWS_OF_EXPLAINING_AWAY and BOUND_ELEMENTS_OF_EXPLAINING_AWAY then
   error('cannot both normalize and bound explaining away')
end

if (CLASS_NLL_CRITERION_TYPE == 'soft_aggregate') and (NORMALIZE_ROWS_OF_CLASS_DICT == true) then
   error('soft-aggregate NLL criterion should be combined with boudned class dict rows, not normalized class dict rows')
end

if (NORMALIZE_ROWS_OF_CLASS_DICT == true) and (BOUND_ROWS_OF_CLASS_DICT == true) then
   error('The rows of the classification dictionary should not be both normalized and bounded')
end



local function duplicate_linear_explaining_away(base_explaining_away, layer_size, num_ista_iterations, use_base_explaining_away, RUN_JACOBIAN_TEST)
   local explaining_away
   if use_base_explaining_away then
      explaining_away = base_explaining_away
   else
      explaining_away = nn.ConstrainedLinear(layer_size[2], layer_size[2], {no_bias = NO_BIAS_ON_EXPLAINING_AWAY, normalized_rows = NORMALIZE_ROWS_OF_EXPLAINING_AWAY, 
									    bounded_elements = BOUND_ELEMENTS_OF_EXPLAINING_AWAY, 
									    filter_weight_matrix = LOCAL_CONNECTIONS_ONLY_EXPLAINING_AWAY_MATRIX}, 
					     RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE, nil, 
					     ((USE_FULL_SCALE_FOR_REPEATED_ISTA_MODULES or RUN_JACOBIAN_TEST) and 1) or 1/num_ista_iterations)
      -- If using Module:updateParameters, distinct gradients should be added to shared parameters; if the gradients are also shared, then the sum of all gradients is added to each copy of the shared parameters, and the gradient step is scaled by the number of copies.  However, if using the optim package, each set of shared parameters is represented only once, and the number of parameterGradients must match the number of parameters, so the gradients must be shared as well.
      if PARAMETER_UPDATE_METHOD == 'optim' then
	 explaining_away:share(base_explaining_away, 'weight', 'bias', 'gradWeight', 'gradBias') 
      elseif PARAMETER_UPDATE_METHOD == 'module' then
	 explaining_away:share(base_explaining_away, 'weight', 'bias')
      else
	 error('unrecognized PARAMETER_UPDATE_METHOD ' .. PARAMETER_UPDATE_METHOD)
      end

      if LOCAL_CONNECTIONS_ONLY_EXPLAINING_AWAY_MATRIX then
	 explaining_away:set_weight_filter_matrix(base_explaining_away:get_weight_filter_matrix())
      end
   end

   return explaining_away
end

local function duplicate_shrink(base_shrink, layer_size, RUN_JACOBIAN_TEST)
   local shrink = nn.ParameterizedShrink(layer_size, FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink) -- EFFICIENCY NOTE: when using non-negative units this could be accomplished more efficiently using an unparameterized, one-sided rectification, just like Glorot, Bordes, and Bengio, along with a non-positive bias in the encoding_feature_extraction_dictionary.  However, both nn.SoftShrink and the shrinkage utility method implemented in kex are two-sided.

   -- If using Module:updateParameters, distinct gradients should be added to shared parameters; if the gradients are also shared, then the sum of all gradients is added to each copy of the shared parameters, and the gradient step is scaled by the number of copies.  However, if using the optim package, each set of shared parameters is represented only once, and the number of parameterGradients must match the number of parameters, so the gradients must be shared as well.
   if PARAMETER_UPDATE_METHOD == 'optim' then
      shrink:share(base_shrink, 'shrink_val', 'grad_shrink_val', 'negative_shrink_val') 
   elseif PARAMETER_UPDATE_METHOD == 'module' then
      shrink:share(base_shrink, 'shrink_val', 'negative_shrink_val') 
   else
      error('unrecognized PARAMETER_UPDATE_METHOD ' .. PARAMETER_UPDATE_METHOD)
   end
   
   if (shrink.shrink_val:storage() ~= base_shrink.shrink_val:storage()) then -- or (shrink.grad_shrink_val:storage() ~= base_shrink.grad_shrink_val:storage()) then
      print('in build_ISTA_iteration, shrink parameters are not shared properly')
      io.read()
   end

   return shrink
end
   
local function duplicate_multiplicative_filter(base_multiplicative_filter, layer_size)
   local multiplicative_filter = nn.MultiplicativeFilter(layer_size, true) -- second argument prevents the duplicate from randomizing the filter
   multiplicative_filter:share(base_multiplicative_filter, 'bias_filter', 'filter_active') -- sharing the filter_active component slaves the duplicates to the original
   return multiplicative_filter
end


-- the input is x [1] (already wrapped in a table)
-- the output is a table of three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
local function build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink)
   local first_ista_seq = nn.Sequential()
   first_ista_seq:add(nn.CopyTable(1,2)) -- split into the transformed input W*x [1], and the untransformed input x [2]

   local first_WX = nn.ParallelTable()
   first_WX:add(encoding_feature_extraction_dictionary) -- transform the input by the transpose dictionary matrix, W*x [1]
   first_WX:add(nn.Identity()) -- preserve the input x [2]
   first_ista_seq:add(first_WX)

   first_ista_seq:add(nn.CopyTable(1,2)) -- split into the subject of the shrink operation z (initially just W*x) [1], the transformed input W*x [2], and the untransformed input x [3]

   local first_shrink_parallel = nn.ParallelTable() -- shrink z; stream is now z = h(W*x) [1], the transformed input W*x [2], and the untransformed input x [3]   
   first_shrink_parallel:add(base_shrink)
   first_shrink_parallel:add(nn.Identity())
   first_shrink_parallel:add(nn.Identity())
   first_ista_seq:add(first_shrink_parallel)

   return first_ista_seq 
end



-- the input is a table of (exactly) three elments: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
-- the output is a table of three (four) elements: the subject of the shrink operation z [1], the transformed input W*x [2], the untransformed input x [3], and possibly the offset shrink [4]
local function build_ISTA_iteration(explaining_away, shrink, RUN_JACOBIAN_TEST, last_iteration, offset_shrink, manually_maintain_explaining_away_diagonal)
   local ista_seq = nn.Sequential()

   if manually_maintain_explaining_away_diagonal then
      -- copy z so we can add it back in after applying the explaining away matrix
      ista_seq:add(nn.CopyTable(1, 2))

      -- apply the explaining away matrix to the first copy of z
      local explaining_away_parallel = nn.ParallelTable() 
      explaining_away_parallel:add(explaining_away) --subtract out the explained input, without the diagonal: S*z [1]
      explaining_away_parallel:add(nn.Identity()) -- preserve z [2]
      explaining_away_parallel:add(nn.Identity()) -- preserve W*x [3]
      explaining_away_parallel:add(nn.Identity()) -- preserve x [4]
      ista_seq:add(explaining_away_parallel)

      -- add S*z to z -- we could also add in W*x at this point, but it seems preferable to maintain compatibility for the original code that explicitly includes the diagonal in the explaining-away matrix
      local add_in_explaining_away_diag_parallel = nn.ParallelDistributingTable() 
      local add_in_explaining_away_diag_seq = nn.Sequential()
      add_in_explaining_away_diag_seq:add(nn.SelectTable{1,2})
      add_in_explaining_away_diag_seq:add(nn.CAddTable()) -- compute z + S*z [1]
      add_in_explaining_away_diag_parallel:add(add_in_explaining_away_diag_seq)
      add_in_explaining_away_diag_parallel:add(nn.SelectTable{3}) -- preserve W*x [2]
      add_in_explaining_away_diag_parallel:add(nn.SelectTable{4}) -- preserve x [3]
      ista_seq:add(add_in_explaining_away_diag_parallel)
   else
      local explaining_away_parallel = nn.ParallelTable() 
      explaining_away_parallel:add(explaining_away) --subtract out the explained input: z + S*z [1]
      explaining_away_parallel:add(nn.Identity()) -- preserve W*x [2]
      explaining_away_parallel:add(nn.Identity()) -- preserve x [3]
      ista_seq:add(explaining_away_parallel)
   end

   -- we need to do this in two stages, since S must be applied only to z, rather than to z + Wx; adding in Wx needs to be the first in the sequence of operations within a ParallelDistributingTable
   local add_in_WX_and_shrink_parallel = nn.ParallelDistributingTable() 
   local add_in_WX_and_shrink_seq = nn.Sequential()
   add_in_WX_and_shrink_seq:add(nn.SelectTable{1,2})
   add_in_WX_and_shrink_seq:add(nn.CAddTable())
   add_in_WX_and_shrink_seq:add(shrink)
   add_in_WX_and_shrink_parallel:add(add_in_WX_and_shrink_seq) -- z = h_theta(z + Wx + S*z) [1]
   add_in_WX_and_shrink_parallel:add(nn.SelectTable{2}) -- preserve W*x [2]
   add_in_WX_and_shrink_parallel:add(nn.SelectTable{3}) -- preserve x [3]
   if CREATE_BUFFER_ON_L1_LOSS and last_iteration then -- create an offset shrink, which can be used rather than the normal shrink output to as input to an L1 loss to ensure there is a buffer around the threshold
      local add_in_WX_and_offset_and_shrink_seq = nn.Sequential()
      add_in_WX_and_offset_and_shrink_seq:add(nn.SelectTable{1,2})
      add_in_WX_and_offset_and_shrink_seq:add(nn.CAddTable())
      add_in_WX_and_offset_and_shrink_seq:add(nn.AddConstant(0, CREATE_BUFFER_ON_L1_LOSS))
      add_in_WX_and_offset_and_shrink_seq:add(offset_shrink)
      add_in_WX_and_shrink_parallel:add(add_in_WX_and_offset_and_shrink_seq)
   end

   ista_seq:add(add_in_WX_and_shrink_parallel)

   return ista_seq
end

-- the input is a table of three (four) elements: the subject of the shrink operation z [1], the transformed input W*x [2], the untransformed input x [3], and possibly the offset shrink [4]
-- the output is a table of three (four) elements: the filtered subject of the shrink operation z [1], the transformed input W*x [2], the untransformed input x [3], and possibly the offset shrink [4]
local function build_multiplicative_filter_layer(this_multiplicative_filter, potentially_create_buffer_on_L1_loss)
   local multiplicative_filter_parallel = nn.ParallelDistributingTable()
   local multiplicative_filter_seq = nn.Sequential()
   multiplicative_filter_seq:add(nn.SelectTable{1})
   multiplicative_filter_seq:add(this_multiplicative_filter)
   
   multiplicative_filter_parallel:add(multiplicative_filter_seq)
   multiplicative_filter_parallel:add(nn.SelectTable{2})
   multiplicative_filter_parallel:add(nn.SelectTable{3})
   if potentially_create_buffer_on_L1_loss and CREATE_BUFFER_ON_L1_LOSS then
      multiplicative_filter_parallel:add(nn.SelectTable{4}) 
   end
   return multiplicative_filter_parallel
end




-- the input is a table of three (four) elments: the subject of the shrink operation z [1], the transformed input W*x [2], the untransformed input x [3], and possibly the offset shrink output [4] 
-- the output is a table of four (five) elements: the subject of the shrink operation z [1], the transformed input W*x [2], the untransformed input x [3], the reconstructed input D*z [4], and possibly the offset shrink output [5]
local function linearly_reconstruct_input(decoding_dictionary)
   local reconstruct_input_parallel = nn.ParallelDistributingTable() -- reconstruct the input from the shrunk code z
   local reconstruct_input_seq = nn.Sequential()
   reconstruct_input_seq:add(nn.SelectTable{1})
   reconstruct_input_seq:add(decoding_dictionary)

   reconstruct_input_parallel:add(nn.SelectTable{1}) -- preserve the subject of the shrink operation z [1]
   reconstruct_input_parallel:add(nn.SelectTable{2}) -- preserve the transformed input W*x [2]
   reconstruct_input_parallel:add(nn.SelectTable{3}) -- preserve the untransformed input x [3]
   reconstruct_input_parallel:add(reconstruct_input_seq) -- reconstruct the input D*z from the shrunk code z [4]
   if CREATE_BUFFER_ON_L1_LOSS then
      reconstruct_input_parallel:add(nn.SelectTable{4}) -- preserve the offset shrink, used to ensure there is a buffer around the threshold [5]
   end
   return reconstruct_input_parallel
end

-- the input is a table of four (five) elments: the layer n code z [1], the transformed input W*x [2], the untransformed layer n-1 input x [3], the reconstructed layer n-1 code r [4], and possibly the layer n-1 offset shrink output [5]
-- the output is a table of three (four) elements: the layer n code z [1], the transformed input W*x [2], the untransformed layer n-1 input x[3], and possibly the layer n-1 offset shrink output [4], as well as the L2 loss criterion
local function build_L2_reconstruction_loss(L2_lambda, criteria_list)
   local combined_loss = nn.ParallelDistributingTable() -- calculate the MSE between the reconstructed layer n-1 code r [2] and untransformed layer n-1 input x [3]; only pass on the layer n code z [1]
   local effective_L2_loss_function   

   local L2_loss_seq = nn.Sequential() -- calculate the MSE between the reconstructed input D*z [4] and the untransformed input x [3]
   L2_loss_seq:add(nn.SelectTable{3,4})
   if DEBUG_L2 or DEBUG_OUTPUT then
      local sequential_zero = nn.Sequential()
      local parallel_zero = nn.ParallelTable() 
      parallel_zero:add(nn.ZeroModule())
      parallel_zero:add(nn.ZeroModule())
      sequential_zero:add(parallel_zero)
      sequential_zero:add(nn.SelectTable{1}) -- a SelectTable is necessary to ensure that the module outputs a single nil, which is ignored by the ParallelDistributingTable, rather than an empty table (i.e., a table of nils), which ParallelDistributingTable incorrectly passes onto its output
      L2_loss_seq:add(sequential_zero) 
   else
      if L2_lambda == nil then
	 error('L2_lambda is incorrectly assigned to be ', L2_lambda, ' with type ', type(L2_lambda))
      end
      effective_L2_loss_function = nn.L2Cost(L2_lambda, 2)
      L2_loss_seq:add(effective_L2_loss_function) -- using this instead of MSECriterion fed through CriterionTable ensures that the gradient is propagated to both inputs
      L2_loss_seq:add(nn.Ignore()) -- don't pass the L2 loss value onto the rest of the network
      table.insert(criteria_list.criteria, effective_L2_loss_function)   
      table.insert(criteria_list.names, 'L2 reconstruction loss')
      print('inserting L2 loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
   end

   -- throw away the reconstructed layer n-1 code r [4]
   combined_loss:add(nn.SelectTable{1}) -- copy the shrunk code z [1] 
   combined_loss:add(nn.SelectTable{2}) -- copy the transformed input W*x [2] 
   combined_loss:add(nn.SelectTable{3}) -- copy the original input x [3]
   if CREATE_BUFFER_ON_L1_LOSS then
      combined_loss:add(nn.SelectTable{5}) -- preserve the offset shrink, used to ensure there is a buffer around the threshold [4]
   end
   combined_loss:add(L2_loss_seq)

   -- rather than using nn.Ignore on the output of the criterion, we could use a SelectTable{1} without a ParallelDistributingTable, which would output a tensor in the forward direction, and send a nil as gradOutput to the criterion in the backwards direction

   return combined_loss, effective_L2_loss_function
end


-- the input is a tensor (*not* a table) consisting of the hidden state z
-- the output is a scalar consisting of the loss
local function build_weighted_L1_criterion(weighted_L1_lambda, pure_L1_lambda)
   local use_true_entropy_loss = true
   local only_apply_scaling_to_softmax = false -- if true, the hidden representation is not uniformly scaled after being subject to L2 normalization but before the entropy or weighted-L1 loss is calculated; rather only the softmax branch is scaled
   local use_l2_norm = USE_L2_NORM_FOR_WEIGHTED_L1 -- fix the L2 norm equal to 1 before calculating the entropy loss
   local use_log_entropy = true -- use log entropy rather than entropy itself; this ensures that high-entropy codes are not subject to a much greater regularization than low-entropy codes

   local narrow_entropy = false
   local append_constant_values = APPEND_CONSTANT_VALUES --{0.1}

   local crit = nn.Sequential()
   -- wrap the input tensor into a table z [1]
   crit:add(nn.IdentityTable()) -- wrap the input tensor in a table z [1]

   -- separate into two streams: L2 normalized input [1], and original input z [2]
   local normalize_input_parallel = nn.ParallelDistributingTable()
   local normalize_input_seq = nn.Sequential()
   normalize_input_seq:add(nn.SelectTable{1})
   if narrow_entropy then
      normalize_input_seq:add(nn.Narrow(2,1,20)) -- EXPERIMENTAL CODE!  Try only applying the entropy loss to the first twenty units
   end
   if append_constant_values then
      normalize_input_seq:add(nn.AppendConstant(append_constant_values))
   end
   normalize_input_seq:add(nn.Abs()) -- this shouldn't actually be necessary, since the units are non-negative
   if use_l2_norm then
      normalize_input_seq:add(nn.NormalizeTensor())
   end
   if not(only_apply_scaling_to_softmax) then
      normalize_input_seq:add(nn.MulConstant(nil, WEIGHTED_L1_SOFTMAX_SCALING * CLASS_DICT_BOUND)) -- this should probably be an input parameter
   end
   if use_true_entropy_loss then
      normalize_input_seq:add(nn.LogSoftMax()) -- do entropy loss rather than weighted-L1
   end
   normalize_input_parallel:add(normalize_input_seq)
   normalize_input_parallel:add(nn.SelectTable{1})
   crit:add(normalize_input_parallel)

   -- separate into three streams: softmax of scaled L2 normalized input [1], L2 normalized input [2], original input [3]
   local softmax_parallel = nn.ParallelDistributingTable()
   local softmax_seq = nn.Sequential()
   softmax_seq:add(nn.SelectTable{1})
   if only_apply_scaling_to_softmax then
      softmax_seq:add(nn.MulConstant(nil, WEIGHTED_L1_SOFTMAX_SCALING * CLASS_DICT_BOUND)) -- this should probably be an input parameter
   end
   if not(use_true_entropy_loss) then
      softmax_seq:add(nn.LogSoftMax())
   end
   softmax_seq:add(nn.Exp())
   softmax_parallel:add(softmax_seq)
   softmax_parallel:add(nn.SelectTable{1})
   softmax_parallel:add(nn.SelectTable{2})
   crit:add(softmax_parallel)

   -- multiply softmax with L2 normalized input to yield: scaled weighted L2 normalized input [1], scaled abs(original input) [2]
   local weight_normed_input_parallel = nn.ParallelDistributingTable()
   local weight_normed_input_seq = nn.Sequential()
   weight_normed_input_seq:add(nn.SelectTable{1,2})
   weight_normed_input_seq:add(nn.SafeCMulTable())
   -- multiply by -1, since the entropy is -p*log(p)
   if use_log_entropy then
      weight_normed_input_seq:add(nn.MulConstant(nil, -1))
      weight_normed_input_seq:add(nn.SumWithinBatch())
      weight_normed_input_seq:add(nn.Log())
      weight_normed_input_seq:add(nn.MulConstant(nil, weighted_L1_lambda)) -- this should match the scaling on the logistic classifier, rather than the normal sparsifying L1 regularizer
   else
      weight_normed_input_seq:add(nn.MulConstant(nil, -1 * weighted_L1_lambda)) -- this should match the scaling on the logistic classifier, rather than the normal sparsifying L1 regularizer
   end
   local entropy_loss_scaling = nn.MulConstant(nil, 1)
   weight_normed_input_seq:add(entropy_loss_scaling)
   if narrow_entropy or append_constant_values or use_log_entropy then
      weight_normed_input_seq:add(nn.SumCriterionModule()) -- EXPERIMENTAL CODE -- returns a number
      weight_normed_input_seq:add(nn.IdentityTensor()) -- EXPERIMENTAL CODE -- wrap in tensor
   end
   local abs_and_scale_seq = nn.Sequential()
   abs_and_scale_seq:add(nn.SelectTable{3})
   abs_and_scale_seq:add(nn.Abs())
   abs_and_scale_seq:add(nn.MulConstant(nil, pure_L1_lambda))
   if narrow_entropy or append_constant_values then
      abs_and_scale_seq:add(nn.SumCriterionModule()) -- EXPERIMENTAL CODE -- returns a number
      abs_and_scale_seq:add(nn.IdentityTensor()) -- EXPERIMENTAL CODE -- wrap in tensor
   end
   weight_normed_input_parallel:add(weight_normed_input_seq)
   weight_normed_input_parallel:add(abs_and_scale_seq)
   crit:add(weight_normed_input_parallel)

   -- add streams together to get a single tensor [1]
   crit:add(nn.CAddTable())

   -- add the elements of the tensor to get the criterion output
   crit:add(nn.SumCriterionModule())
   --crit:add(nn.L1CriterionModule(nn.L1Cost()))

   function crit:reset_entropy_scale_factor(new_constant) entropy_loss_scaling:reset_constant(new_constant) end

   return crit
end






-- the input is a tensor (*not* a table) consisting of the hidden state z
-- the output is a scalar consisting of the loss
-- rather than using the softmax to compute the probability, just do an L1 normalization
local function build_hard_entropy_criterion(weighted_L1_lambda, pure_L1_lambda) 
   local append_constant_values = {0.1} --{0.05} --{0.1}
   local add_constant_value = (1/400) -- note that the first-order Taylor expansion of the exponential around 0 is x + 1.  Using a smaller additive constant seems similar to using a larger softmax scaling factor.  Note that, when computing the entropy, the contribution of a single unit is weighted against approximately 400 units with the add_constant_value.  If the largest value attainable by a single unit is about 1, then add_constant_value should have order less than 1/400.  

   local crit = nn.Sequential()
   -- wrap the input tensor into a table z [1]
   crit:add(nn.IdentityTable()) -- wrap the input tensor in a table z [1]

   -- separate into two streams: L2 normalized input [1], and original input z [2]
   local entropy_parallel = nn.ParallelDistributingTable()
   local entropy_seq = nn.Sequential()
   entropy_seq:add(nn.SelectTable{1})
   if append_constant_values then
      entropy_seq:add(nn.AppendConstant(append_constant_values))
   end
   if add_constant_value then
      entropy_seq:add(nn.AddConstant(nil, add_constant_value))
   end
   entropy_seq:add(nn.Abs()) -- this shouldn't actually be necessary, since the units are non-negative
   entropy_seq:add(nn.NormalizeTensorL1())
   entropy_seq:add(nn.SafeEntropy()) -- returns -p*log(p) for each input (already normalized)
   if append_constant_values then
      entropy_seq:add(nn.SumCriterionModule()) -- returns a number
      entropy_seq:add(nn.IdentityTensor()) -- wrap in tensor
   end
   entropy_seq:add(nn.MulConstant(nil, weighted_L1_lambda)) -- this should match the scaling on the logistic classifier, rather than the normal sparsifying L1 regularizer
   local entropy_loss_scaling = nn.MulConstant(nil, 1)
   entropy_seq:add(entropy_loss_scaling) 

   local abs_and_scale_seq = nn.Sequential()
   abs_and_scale_seq:add(nn.SelectTable{1})
   abs_and_scale_seq:add(nn.Abs())
   if append_constant_values then
      abs_and_scale_seq:add(nn.SumCriterionModule()) -- returns a number
      abs_and_scale_seq:add(nn.IdentityTensor()) -- wrap in tensor
   end
   abs_and_scale_seq:add(nn.MulConstant(nil, pure_L1_lambda))

   entropy_parallel:add(entropy_seq)
   entropy_parallel:add(abs_and_scale_seq)
   crit:add(entropy_parallel)

   -- add streams together to get a single tensor [1]
   crit:add(nn.CAddTable())

   -- add the elements of the tensor to get the criterion output
   crit:add(nn.SumCriterionModule())

   function crit:reset_entropy_scale_factor(new_constant) entropy_loss_scaling:reset_constant(new_constant) end

   return crit
end








-- Use two different values for L1 on disjoint subsets of the hidden units
-- the input is a tensor (*not* a table) consisting of the hidden state z
-- the output is a scalar consisting of the loss
function build_heterogeneous_L1_criterion(layer_size, first_L1_lambda, second_L1_lambda)
   local crit = nn.Sequential()
   local split_sparsifying_concat = nn.Concat(1) 
   local first_split = nn.Sequential()
   local num_putative_categorical_units = math.ceil(0.2 * layer_size[2])
   first_split:add(nn.Narrow(1,1,num_putative_categorical_units))
   first_split:add(nn.L1CriterionModule(nn.L1Cost(), first_L1_lambda))
   first_split:add(nn.IdentityTensor()) -- nn.Concat only combines tensors, whereas the criterion return scalars, so wrap them up, so we can then add them for unit testing
   local second_split = nn.Sequential()
   second_split:add(nn.Narrow(1,1 + num_putative_categorical_units,layer_size[2] - num_putative_categorical_units)) 
   second_split:add(nn.L1CriterionModule(nn.L1Cost(), second_L1_lambda))
   second_split:add(nn.IdentityTensor()) -- nn.Concat only combines tensors, whereas the criterion return scalars, so wrap them up, so we can then add them for unit testing
   split_sparsifying_concat:add(first_split)
   split_sparsifying_concat:add(second_split)
   
   crit:add(nn.Transpose_rp()) -- ensures that the first dimension should be split, even if the network is processing a batch, for which the input iterations are usually along the first dimension
   crit:add(split_sparsifying_concat)
   crit:add(nn.Sum(1)) -- add up the losses from the two splits; returns a scalar, since its summing up a one-dimensional tensor
   crit:add(nn.SafeIdentity()) -- make sure that the nn.Sum() receives a tensor as a gradOutput, rather than a nil, even though its gradInput will be ignored by the L1CriterionModules
   return crit
end


-- Use (\sum_i |z_i|) / (\sum_i z_i^2)^(1/2) as the regularizer (L1 / L2) in addition to the normal L1 regularizer
-- the input is a tensor (*not* a table) consisting of the hidden state z
-- the output is a scalar consisting of the loss
function build_L1_over_L2_criterion(pure_L1_lambda, L1_over_L2_lambda)
   crit = nn.Sequential()
   local combined_sparsifying_concat = nn.Concat(1) 
   local pure_L1_crit = nn.Sequential()
   pure_L1_crit:add(nn.L1CriterionModule(nn.L1Cost(), pure_L1_lambda))
   pure_L1_crit:add(nn.IdentityTensor()) -- nn.Concat only combines tensors, whereas the criterion return scalars, so wrap them up, so we can then add them for unit testing
   local L1_over_L2_crit = nn.Sequential()
   L1_over_L2_crit:add(nn.L1CriterionModule(nn.L1OverL2Cost(), L1_over_L2_lambda)) -- 0.3
   L1_over_L2_crit:add(nn.IdentityTensor()) -- nn.Concat only combines tensors, whereas the criterion return scalars, so wrap them up, so we can then add them for unit testing
   combined_sparsifying_concat:add(pure_L1_crit)
   combined_sparsifying_concat:add(L1_over_L2_crit)
   
   crit:add(combined_sparsifying_concat)
   crit:add(nn.Sum(1)) -- add up the losses from the two splits; returns a scalar, since its summing up a one-dimensional tensor
   crit:add(nn.SafeIdentity()) -- make sure that the nn.Sum() receives a tensor as a gradOutput, rather than a nil, even though its gradInput will be ignored by the L1CriterionModules
   return crit
end


-- the input is a table of three (four) elements: the subject of the shrink operation z [1], the transformed input W*x [2], the original input x [3], the possibly the offset shrink output [4]
-- the output is a table of (exactly) three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the original input x [3]
local function build_sparsifying_loss(sparsifying_criterion_module, criteria_list, use_offset_shrink_output)
   local apply_L1_norm = nn.ParallelDistributingTable()
   local scaled_L1_norm = nn.Sequential() -- scale the code [1], calculate its L1 norm, and throw away the output
   if use_offset_shrink_output then
      scaled_L1_norm:add(nn.SelectTable{4}) 
   else
      scaled_L1_norm:add(nn.SelectTable{1}) 
   end
   if DEBUG_L1 or DEBUG_OUTPUT then
      scaled_L1_norm:add(nn.ZeroModule())
   else
      --local sparsifying_criterion_module = nn.L1CriterionModule(sparsifying_loss_function, sparsifying_lambda) -- the L1CriterionModule, rather than the wrapped criterion, produces the correct scaled error
      --local sparsifying_criterion_module = ParameterizedL1Cost(layer_size, sparsifying_lambda, desired_criterion_value, lagrange_multiplier_learning_rate_scaling_factor)
      scaled_L1_norm:add(sparsifying_criterion_module) -- also compute the L1 norm on the code
      scaled_L1_norm:add(nn.Ignore()) -- don't pass the L1 loss value onto the rest of the network
      table.insert(criteria_list.criteria, sparsifying_criterion_module) -- make sure that we consider the sparsifying_loss_function when evaluating the total loss
      table.insert(criteria_list.names, 'sparsifying loss')
      print('inserting sparsifying loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
   end
   apply_L1_norm:add(nn.SelectTable{1}) -- pass the code [1] through unchanged for further processing   
   apply_L1_norm:add(nn.SelectTable{2}) -- pass the transformed input W*x [2] through unchanged for further processing   
   apply_L1_norm:add(nn.SelectTable{3}) -- pass the original input x [3] through unchanged for further processing   
   apply_L1_norm:add(scaled_L1_norm) -- since we add scaled_L1_norm without a SelectTable, it receives 

   return apply_L1_norm 
end

-- the input is a table of two elements: the output of the shrink operation z [1], and the original input x [2]
-- the output is a table of three elements: the subject of the pooling operation s [1], the preserved input z [2], and the original input x [3]
local function build_pooling(encoding_pooling_dictionary)
   local pool_seq = nn.Sequential()
   pool_seq:add(nn.CopyTable(1, 2)) -- split into the output to the rest of the chain z [1] and the preserved input z [2], used to calculate the reconstruction error, and the original input x [3]

   local pool_features_parallel = nn.ParallelTable() -- compute s = sqrt(Q*z^2)
   local pooling_transformation_seq = nn.Sequential()
   pooling_transformation_seq:add(nn.Square()) -- EFFICIENCY NOTE: nn.Square is almost certainly more efficient than Power(2), but it computes gradInput incorrectly on 1-D inputs; specifically, it fails to multiply the gradInput by two.  This can and should be corrected, but doing so will require modifying c code.
   pooling_transformation_seq:add(encoding_pooling_dictionary)
   local safe_sqrt_const = 1e-10
   pooling_transformation_seq:add(nn.AddConstant(encoding_pooling_dictionary.weight:size(1), safe_sqrt_const)) -- ensures that we never compute the gradient of sqrt(0), so long as encoding_pooling_dictionary is non-negative with no bias.  Keep in mind that we take the square root of this quantity, so even a seemingly small number like 1e-5 becomes a relatively large constant of 0.00316.  At the same time, the gradient of the square root divides by the square root, and so becomes huge as the argument of the square root approaches zero

   -- Note that backpropagating gradients through nn.Sqrt will generate NANs if any of the inputs are exactly zero.  However, this is correct behavior.  We should ensure that no input to the square root is exactly zero, perhaps by bounding the encoding_pooling_dictionary below by a number greater than zero.
   pooling_transformation_seq:add(nn.Sqrt()) -- NORMAL VERSION
   pooling_transformation_seq:add(nn.AddConstant(encoding_pooling_dictionary.weight:size(1), -math.sqrt(safe_sqrt_const))) -- we add 1e-4 before the square root and subtract (1e-4)^1/2 after the square root, so zero is still mapped to zero, but the gradient contribution is bounded above by 100

   pool_features_parallel:add(pooling_transformation_seq) -- s = sqrt(Q*z^2) [1]
   pool_features_parallel:add(nn.Identity()) -- preserve z [2]
   pool_features_parallel:add(nn.Identity()) -- preserve x [3]
   pool_seq:add(pool_features_parallel) 

   return pool_seq
end


local function build_loss_tester(num_inputs, tested_inputs, criteria_list)
   print('ADDING LOSS TESTER!')
   local test_pdt = nn.ParallelDistributingTable()
   
   for i = 1,num_inputs do
      test_pdt:add(nn.SelectTable{i})
   end

   if tested_inputs then 
      for k,v in ipairs(tested_inputs) do
	 local test_L2_loss = nn.L2Cost(math.random(), 1)
	 table.insert(criteria_list.criteria, test_L2_loss) -- make sure that we consider the L2_position_loss when evaluating the total loss
	 table.insert(criteria_list.names, 'test L2 loss ' .. v)
	 print('inserting test L2 loss ' .. v .. ' into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
	 
	 local test_seq = nn.Sequential()
	 test_seq:add(nn.SelectTable{v})
	 test_seq:add(test_L2_loss)
	 test_seq:add(nn.Ignore())
	 test_pdt:add(test_seq)
      end
   end

   return test_pdt
end


-- creates two copies of the base_decoding_feature_extraction_dictionary, the second of which has the weight matrix transposed
-- also returns a straightened copy of the transposed copy, which copies both the weight and bias *and* the gradWeight
local function duplicate_decoding_feature_extraction_dictionary(base_decoding_feature_extraction_dictionary, layer_size, make_transpose, RUN_JACOBIAN_TEST)
   local decoding_feature_extraction_dictionary_copy = nn.ConstrainedLinear(layer_size[2], layer_size[1], {no_bias = true, normalized_columns = true}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
   local decoding_feature_extraction_dictionary_transpose, decoding_feature_extraction_dictionary_transpose_straightened_copy -- straightened_copy is required for jacobian testing of the module update method
   if make_transpose then
      decoding_feature_extraction_dictionary_transpose = nn.ConstrainedLinear(layer_size[1], layer_size[2], {no_bias = true, normalized_columns = true}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE)

      decoding_feature_extraction_dictionary_transpose.weight:set(base_decoding_feature_extraction_dictionary.weight:t())
      decoding_feature_extraction_dictionary_transpose.accUpdateGradParameters = decoding_feature_extraction_dictionary_transpose.sharedAccUpdateGradParameters
   end
   
   -- If using Module:updateParameters, distinct gradients should be added to shared parameters; if the gradients are also shared, then the sum of all gradients is added to each copy of the shared parameters, and the gradient step is scaled by the number of copies.  However, if using the optim package, each set of shared parameters is represented only once, and the number of parameterGradients must match the number of parameters, so the gradients must be shared as well.
   if PARAMETER_UPDATE_METHOD == 'optim' then
      decoding_feature_extraction_dictionary_copy:share(base_decoding_feature_extraction_dictionary, 'weight', 'bias', 'gradWeight', 'gradBias') 
      if make_transpose then
	 decoding_feature_extraction_dictionary_transpose.gradWeight:set(base_decoding_feature_extraction_dictionary.gradWeight:t()) -- don't share bias, but this is OK since bias is set to zero
      end
   elseif PARAMETER_UPDATE_METHOD == 'module' then
      decoding_feature_extraction_dictionary_copy:share(base_decoding_feature_extraction_dictionary, 'weight', 'bias')

      if make_transpose and RUN_JACOBIAN_TEST then
	 decoding_feature_extraction_dictionary_transpose_straightened_copy = nn.ConstrainedLinear(layer_size[2],layer_size[1], {no_bias = true, normalized_columns = true}, 
												   RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
	 decoding_feature_extraction_dictionary_transpose_straightened_copy:share(base_decoding_feature_extraction_dictionary, 'weight', 'bias')
	 decoding_feature_extraction_dictionary_transpose_straightened_copy.gradWeight:set(decoding_feature_extraction_dictionary_transpose.gradWeight:t()) -- don't share bias, but this is OK since bias is set to zero
      end
   else
      error('unrecognized PARAMETER_UPDATE_METHOD ' .. PARAMETER_UPDATE_METHOD)
   end

   return decoding_feature_extraction_dictionary_copy, decoding_feature_extraction_dictionary_transpose, decoding_feature_extraction_dictionary_transpose_straightened_copy
end

-- the input is a table of three elements: the subject of the pooling operation s [1], the preserved shrink output z [2], and the original input x [3]
-- the output is a table of two elements: the subject of the pooling operation s [1] and the original input x [2]
local function build_pooling_L2_loss(decoding_pooling_dictionary, decoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary_transpose, mask_sparsifying_module, lambdas, criteria_list)
   -- the L2 reconstruction error and the L2 position unit magnitude both depend upon the denominator lambda + (P*s)^2, so calculate them together in a single function
   --lambdas include: lambdas.pooling_L2_shrink_reconstruction_lambda, lambdas.pooling_L2_orig_reconstruction_lambda, lambdas.pooling_L2_shrink_position_unit_lambda, lambdas.pooling_L2_orig_position_unit_lambda,

   local USE_SQRT_POOLING_RECONSTRUCTION = true

   local pooling_L2_loss_seq = nn.Sequential()
   -- split into the output from the module s = sqrt(Q*z^2) [1], the basis of the denominator of the losses s [2], the preserved shrink output z [3], the original input x [4], and the basis for the projected original input D^t*x [5]
   pooling_L2_loss_seq:add(nn.CopyTable(1, 2)) -- copy the subject of the pooling operation s [1] to [2]
   pooling_L2_loss_seq:add(nn.CopyTable(4, 2)) -- copy the original input x [4] to [5]

   -- split into s [1], P*s [2], z [3], the original input x [4], and the projected original input D^t*x [5]
   local reconstruction_parallel = nn.ParallelTable() -- compute the reconstruction of the input P*s
   reconstruction_parallel:add(nn.Identity()) -- preserve the output s = sqrt(Q*z^2) [1]
   reconstruction_parallel:add(decoding_pooling_dictionary) -- compute the reconstruction P*s [2]
   reconstruction_parallel:add(nn.Identity()) -- preserve the shrink output z [3]
   reconstruction_parallel:add(nn.Identity()) -- preserve the original input x [4]
   reconstruction_parallel:add(decoding_feature_extraction_dictionary_transpose) -- compute D^t *x [5]
   pooling_L2_loss_seq:add(reconstruction_parallel)

   -- compute the sparsifying regularizer on the pooling mask P*s [2]; output is still s [1], P*s [2], z [3], the original input x [4], and the projected original input D^t*x [5]
   local sparsifying_loss_seq = nn.Sequential()
   sparsifying_loss_seq:add(nn.SelectTable{2})
   if DEBUG_OUTPUT then 
      sparsifying_loss_seq:add(nn.ZeroModule())
   else
      sparsifying_loss_seq:add(mask_sparsifying_module) -- also compute the sparsifying norm on the code
      table.insert(criteria_list.criteria, mask_sparsifying_module) -- make sure that we consider the sparsifying_loss_function when evaluating the total loss
      table.insert(criteria_list.names, 'sparsifying mask loss')
      print('inserting sparsifying mask loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
   end
   sparsifying_loss_seq:add(nn.Ignore()) -- don't pass the sparsifying loss value onto the rest of the network
   local sparsifying_loss_parallel = nn.ParallelDistributingTable() -- we could probably use a CopyTable with zero repeats here
   sparsifying_loss_parallel:add(nn.SelectTable{1}) -- s [1]
   sparsifying_loss_parallel:add(nn.SelectTable{2}) -- P*s [2] -- NORMAL VERSION!!!  
   --local double_rec_seq = nn.Sequential() -- DEBUG ONLY!!!
   --double_rec_seq:add(nn.SelectTable{2}) -- DEBUG ONLY!!!
   --double_rec_seq:add(nn.MulConstant(decoding_pooling_dictionary.weight:size(1), 1.5)) -- DEBUG ONLY!!!
   --sparsifying_loss_parallel:add(double_rec_seq) -- DEBUG ONLY!!!
   sparsifying_loss_parallel:add(nn.SelectTable{3}) -- z [3]
   sparsifying_loss_parallel:add(nn.SelectTable{4}) -- x [4] 
   sparsifying_loss_parallel:add(nn.SelectTable{5}) -- D^t*x [5] 
   sparsifying_loss_parallel:add(sparsifying_loss_seq) -- produces no output
   pooling_L2_loss_seq:add(sparsifying_loss_parallel)


   -- the output from the module s [1], the original input x [2], the numerator of the shrink reconstruction loss lambda_ratio*z [3], the numerator of the shrink position loss z*(P*s) [4], 
   -- the numerator of the original reconstruction loss z*(P*s)^2 [5], the numerator of the original position loss D^t*x*(P*s) [6], and the denominator of the loss lambda_ratio + (P*s)^2 [7] 
   -- lambda_ratio arises from minimizing the sum of the shrink reconstruction and the L2 position unit error with respect to the position units, in order to find the feedforward value of the position units
   local lambda_ratio = math.min((lambdas.pooling_L2_shrink_position_unit_lambda + lambdas.pooling_L2_orig_position_unit_lambda) / (lambdas.pooling_L2_shrink_reconstruction_lambda + lambdas.pooling_L2_orig_reconstruction_lambda), 1e5) -- bound the lambda_ratio, so signals and gradients remain finite even if L2_reconstruction_lambda = 0
   if lambda_ratio ~= lambda_ratio then lambda_ratio = 1 end-- if lambda_ratio == nan, set it to 1
   local construct_shrink_rec_numerator_seq = nn.Sequential() -- lambda_ratio * z
   construct_shrink_rec_numerator_seq:add(nn.SelectTable{3})
   construct_shrink_rec_numerator_seq:add(nn.MulConstant(decoding_pooling_dictionary.weight:size(1), lambda_ratio))  -- first argument just sets the size of the output, rather than the constant
   local construct_shrink_pos_numerator_seq = nn.Sequential() -- z*(P*s)
   construct_shrink_pos_numerator_seq:add(nn.SelectTable{2,3}) 
   construct_shrink_pos_numerator_seq:add(nn.SafeCMulTable())
   local construct_orig_rec_numerator_seq = nn.Sequential() -- z*(P*s)^2 
   construct_orig_rec_numerator_seq:add(nn.SelectTable{2,3})
   local square_Ps_for_orig_rec_numerator = nn.ParallelTable()
   square_Ps_for_orig_rec_numerator:add(nn.Square()) 
   square_Ps_for_orig_rec_numerator:add(nn.Identity())
   construct_orig_rec_numerator_seq:add(square_Ps_for_orig_rec_numerator)
   construct_orig_rec_numerator_seq:add(nn.SafeCMulTable())
   local construct_orig_pos_numerator_seq = nn.Sequential() -- D^t*x*(P*s)
   construct_orig_pos_numerator_seq:add(nn.SelectTable{2,5}) 
   construct_orig_pos_numerator_seq:add(nn.SafeCMulTable())
   local construct_denominator_seq = nn.Sequential() -- lambda_ratio + (P*s)^2
   construct_denominator_seq:add(nn.SelectTable{2}) 
   construct_denominator_seq:add(nn.Square())
   construct_denominator_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), lambda_ratio)) 

   -- alternate terms used when the position loss is ||sqrt(P*s)*y||^2 rather than ||y||^2, where y is the L2-normalized position units
   local construct_alt_pos_numerator_seq = nn.Sequential() -- z*(P*s)^0.5 
   construct_alt_pos_numerator_seq:add(nn.SelectTable{2,3})
   local safe_sqrt_seq = nn.Sequential()
   local safe_sqrt_const = 1e-10 --1e-7
   safe_sqrt_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), safe_sqrt_const)) -- ensures that we never compute the gradient of sqrt(0)
   safe_sqrt_seq:add(nn.Sqrt())
   safe_sqrt_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), -math.sqrt(safe_sqrt_const))) -- we add 1e-4 before the square root and subtract (1e-4)^1/2 after the square root, so zero is still mapped to zero, but the gradient contribution is bounded above by 100
   local sqrt_Ps_for_alt_pos_numerator = nn.ParallelTable()
   sqrt_Ps_for_alt_pos_numerator:add(safe_sqrt_seq) 
   sqrt_Ps_for_alt_pos_numerator:add(nn.Identity())
   construct_alt_pos_numerator_seq:add(sqrt_Ps_for_alt_pos_numerator)
   construct_alt_pos_numerator_seq:add(nn.SafeCMulTable())
   local construct_alt_denominator_seq = nn.Sequential() -- lambda_ratio + (P*s)
   construct_alt_denominator_seq:add(nn.SelectTable{2}) 
   construct_alt_denominator_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), lambda_ratio)) 

   local construct_num_denom_parallel = nn.ParallelDistributingTable('construct_num_denom_parallel') -- put it all together
   construct_num_denom_parallel:add(nn.SelectTable{1}) -- preserve the output s = sqrt(Q*z^2) [1]
   construct_num_denom_parallel:add(nn.SelectTable{4}) -- preserve the original input x [2]
   construct_num_denom_parallel:add(construct_shrink_rec_numerator_seq) -- compute the shrink reconstruction numerator lambda_ratio * z [3]
   construct_num_denom_parallel:add(construct_shrink_pos_numerator_seq) -- compute the shrink position numerator z*(P*s) [4]
   construct_num_denom_parallel:add(construct_orig_rec_numerator_seq) -- compute the original input reconstruction numerator z*(P*s)^2 [5]
   construct_num_denom_parallel:add(construct_orig_pos_numerator_seq) -- compute the original input position numerator D^t*x*(P*s) [6]
   construct_num_denom_parallel:add(construct_denominator_seq) -- compute the denominator lambda_ratio + (P*s)^2 [7]

   construct_num_denom_parallel:add(construct_alt_pos_numerator_seq) -- compute the shrink position numerator with modified L2 position loss z*(P*s)^0.5 [8]
   construct_num_denom_parallel:add(construct_alt_denominator_seq) -- compute the denominator with modified L2 position loss lambda_ratio + (P*s) [9]
   pooling_L2_loss_seq:add(construct_num_denom_parallel)
   

   -- Build the four loss functions.  They will all be plugged into a parallel distributing table, and so can begin with a SelectTable

   -- compute the shrink reconstruction loss ||z - (z*(P*s)^2)/(lambda_ratio + (P*s)^2)||^2 = ||lambda_ratio * z / (lambda_ratio + (P*s)^2)||^2
   local compute_shrink_reconstruction_loss_seq = nn.Sequential()
   if USE_SQRT_POOLING_RECONSTRUCTION then 
      compute_shrink_reconstruction_loss_seq:add(nn.SelectTable{3,9}) -- DEBUG ONLY!!! SHOULD BE {3,7} to use normal losses
   else
      compute_shrink_reconstruction_loss_seq:add(nn.SelectTable{3,7}) -- NORMAL VERSION!!!
   end
   compute_shrink_reconstruction_loss_seq:add(nn.CDivTable())
   if DEBUG_OUTPUT then 
      compute_shrink_reconstruction_loss_seq:add(nn.ZeroModule())
   else
      local L2_shrink_reconstruction_loss = nn.L2Cost(lambdas.pooling_L2_shrink_reconstruction_lambda, 1)
      table.insert(criteria_list.criteria, L2_shrink_reconstruction_loss) -- make sure that we consider the L2_shrink_reconstruction_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 shrink reconstruction loss')
      print('inserting L2 pooling shrink reconstruction loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')

      compute_shrink_reconstruction_loss_seq:add(L2_shrink_reconstruction_loss)
   end
   compute_shrink_reconstruction_loss_seq:add(nn.Ignore)
   

   -- compute the original reconstruction loss ||x - D* (z*(P*s)^2 / (lambda_ratio + (P*s)^2))||^2 
   local compute_orig_reconstruction_loss_seq = nn.Sequential()
   compute_orig_reconstruction_loss_seq:add(nn.SelectTable{2,5,7})
   local divide_input_rec_seq = nn.Sequential()
   divide_input_rec_seq:add(nn.SelectTable{2,3})
   divide_input_rec_seq:add(nn.CDivTable())
   divide_input_rec_seq:add(decoding_feature_extraction_dictionary)
   local divide_input_rec_parallel = nn.ParallelDistributingTable('divide_input_rec')
   divide_input_rec_parallel:add(nn.SelectTable{1})
   divide_input_rec_parallel:add(divide_input_rec_seq)
   compute_orig_reconstruction_loss_seq:add(divide_input_rec_parallel)
   if DEBUG_OUTPUT then 
      local parallel_zero = nn.ParallelTable() 
      parallel_zero:add(nn.ZeroModule())
      parallel_zero:add(nn.ZeroModule())
      compute_orig_reconstruction_loss_seq:add(parallel_zero)
      compute_orig_reconstruction_loss_seq:add(nn.SelectTable{1}) -- a SelectTable is necessary to ensure that the module outputs a single nil, which is ignored by the ParallelDistributingTable, rather than an empty 
   else
      local L2_orig_reconstruction_loss = nn.L2Cost(lambdas.pooling_L2_orig_reconstruction_lambda, 2)
      table.insert(criteria_list.criteria, L2_orig_reconstruction_loss) -- make sure that we consider the L2_orig_reconstruction_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 orig reconstruction loss')
      print('inserting L2 pooling orig reconstruction loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      
      compute_orig_reconstruction_loss_seq:add(L2_orig_reconstruction_loss)
   end
   compute_orig_reconstruction_loss_seq:add(nn.Ignore)


   -- compute the shrink position loss ||z*(P*s) / (lambda_ratio + (P*s)^2)||^2
   local compute_shrink_position_loss_seq = nn.Sequential()
   if USE_SQRT_POOLING_RECONSTRUCTION then
      compute_shrink_position_loss_seq:add(nn.SelectTable{8,9}) -- DEBUG ONLY!!!
   else
      compute_shrink_position_loss_seq:add(nn.SelectTable{4,7}) -- NORMAL VERSION
   end
   local compute_shrink_position_units = nn.CDivTable()
   compute_shrink_position_loss_seq:add(compute_shrink_position_units)
   if DEBUG_OUTPUT then 
      compute_shrink_position_loss_seq:add(nn.ZeroModule())
   else
      local L2_shrink_position_loss = nn.L2Cost(lambdas.pooling_L2_shrink_position_unit_lambda, 1)
      table.insert(criteria_list.criteria, L2_shrink_position_loss) -- make sure that we consider the L2_shrink_position_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 shrink position loss')
      print('inserting L2 pooling shrink position loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      
      compute_shrink_position_loss_seq:add(L2_shrink_position_loss)
   end
   compute_shrink_position_loss_seq:add(nn.Ignore)


   -- compute the original input position loss ||D^t*x*(P*s) / (lambda_ratio + (P*s)^2)||^2
   local compute_orig_position_loss_seq = nn.Sequential()
   compute_orig_position_loss_seq:add(nn.SelectTable{6,7})
   local compute_orig_position_units = nn.CDivTable()
   compute_orig_position_loss_seq:add(compute_orig_position_units)
   if DEBUG_OUTPUT then 
      compute_orig_position_loss_seq:add(nn.ZeroModule())
   else
      local L2_orig_position_loss = nn.L2Cost(lambdas.pooling_L2_orig_position_unit_lambda, 1)
      table.insert(criteria_list.criteria, L2_orig_position_loss) -- make sure that we consider the L2_orig_position_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 orig position loss')
      print('inserting L2 pooling orig position loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      
      compute_orig_position_loss_seq:add(L2_orig_position_loss)
   end
   compute_orig_position_loss_seq:add(nn.Ignore)


   -- put everything together
   local compute_loss_parallel = nn.ParallelDistributingTable('compute_loss_parallel') 
   compute_loss_parallel:add(nn.SelectTable{1}) -- preserve the output s = sqrt(Q*z^2) [1]
   compute_loss_parallel:add(nn.SelectTable{2}) -- preserve the original input x [2]
   compute_loss_parallel:add(compute_shrink_reconstruction_loss_seq) -- compute the shrink reconstruction loss ||lambda_ratio * z / (lambda_ratio + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_orig_reconstruction_loss_seq) -- compute the original reconstruction loss ||x - z*(P*s)^2 / (lambda_ratio + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_shrink_position_loss_seq) -- compute the position loss ||z*(P*s) / (lambda_ratio + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_orig_position_loss_seq) -- compute the original input position loss ||D^t*x*(P*s) / (lambda_ratio + (P*s)^2)||^2 [-]
   pooling_L2_loss_seq:add(compute_loss_parallel)

   local debug_modules = {pooling_L2_loss_seq = pooling_L2_loss_seq, 
			  compute_shrink_position_units = compute_shrink_position_units, 
			  compute_orig_position_units = compute_orig_position_units, 
			  compute_shrink_reconstruction_loss_seq = compute_shrink_reconstruction_loss_seq, 
			  compute_orig_reconstruction_loss_seq = compute_orig_reconstruction_loss_seq,
			  compute_shrink_position_loss_seq = compute_shrink_position_loss_seq, 
			  compute_orig_position_loss_seq = compute_orig_position_loss_seq, 
			  construct_shrink_rec_numerator_seq = construct_shrink_rec_numerator_seq,
			  construct_shrink_pos_numerator_seq = construct_shrink_pos_numerator_seq, 
			  construct_orig_rec_numerator_seq = construct_orig_rec_numerator_seq, 
			  construct_orig_pos_numerator_seq = construct_orig_pos_numerator_seq, 
			  construct_denominator_seq = construct_denominator_seq}
   return pooling_L2_loss_seq, debug_modules
end



-- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer, repair_interval, manually_maintain_explaining_away_diagonal
-- lambdas consists of an array of arrays, so it's difficult to make an off-by-one-error when initializing
-- build a stack of reconstruction-pooling layers, followed by a classfication dictionary and associated criterion
function build_recpool_net(layer_size, lambdas, classification_criterion_lambda, lagrange_multiplier_targets, lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, data_set, RUN_JACOBIAN_TEST)
   -- lambdas: {ista_L2_reconstruction_lambda, ista_L1_lambda, pooling_L2_shrink_reconstruction_lambda, pooling_L2_orig_reconstruction_lambda, pooling_L2_shrink_position_unit_lambda, pooling_L2_orig_position_unit_lambda, pooling_output_cauchy_lambda, pooling_mask_cauchy_lambda}
   -- disable_pooling specifies that the pooling layers should be created (otherwise all of the test code that depends on their existence will fail), but that they should not be added to model = nn.Sequential().  As a result, they should not slow down the running time at all.  

   if NORMALIZE_ROWS_OF_EXPLAINING_AWAY and not(recpool_config_prefs.manually_maintain_explaining_away_diagonal) then
      error('trying to normalize rows of explaining away while explicitly including the diagonal in the explaining away matrix')
   end


   local criteria_list = {criteria = {}, names = {}} -- list of all criteria and names comprising the loss function.  These are necessary to run the Jacobian unit test forwards
   RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST or false
   
   recpool_config_prefs.shrink_style = recpool_config_prefs.shrink_style or 'ParameterizedShrink'

   local model = nn.Sequential()
   local layer_list = {} -- an array of the component layers for easy access
   -- size of the classification dictionary depends upon whether we're using pooling or not
   local classification_dictionary
   local num_cols_class_dict
   if not(recpool_config_prefs.disable_pooling) then
      num_cols_class_dict = layer_size[#layer_size-1]
   else  -- if we're not pooling, then the classification dictionary needs to map from the feature extraction layer to the classes, rather than from the pooling layer to the classes
      num_cols_class_dict = layer_size[#layer_size-2]
   end

   if NORMALIZE_ROWS_OF_CLASS_DICT or BOUND_ROWS_OF_CLASS_DICT then
      classification_dictionary = nn.ConstrainedLinear(num_cols_class_dict, layer_size[#layer_size], 
						       {normalized_rows = NORMALIZE_ROWS_OF_CLASS_DICT, bounded_elements = BOUND_ROWS_OF_CLASS_DICT},
						       RUN_JACOBIAN_TEST, ((RUN_JACOBIAN_TEST and 1) or CLASS_DICT_GRAD_SCALING))
   else
      classification_dictionary = nn.Linear(num_cols_class_dict, layer_size[#layer_size])
   end

   local this_class_nll_criterion
   if UNSUPERVISED_CLASSIFICATION or GROUP_SPARISTY_TEN_FIXED_GROUPS then
      this_class_nll_criterion = nn.L1Cost()
   elseif CLASS_NLL_CRITERION_TYPE == 'soft' then
      this_class_nll_criterion = nn.SoftClassNLLCriterion()
   elseif CLASS_NLL_CRITERION_TYPE == 'soft_aggregate' then
      this_class_nll_criterion = nn.SoftClassNLLCriterion(true)
   elseif CLASS_NLL_CRITERION_TYPE == 'hinge' then
      this_class_nll_criterion = nn.HingeClassNLLCriterion()
   elseif not(CLASS_NLL_CRITERION_TYPE) then
      this_class_nll_criterion = nn.ClassNLLCriterion()
   else
      error('Did not recognized desired CLASS_NLL_CRITERION_TYPE ' .. CLASS_NLL_CRITERION_TYPE)
   end
   this_class_nll_criterion.sizeAverage = false -- ABSOLUTELY CRITICAL to ensure that the gradients from the classification loss alone are not scaled down in proportion to the minibatch size
   local classification_criterion = nn.L1CriterionModule(this_class_nll_criterion, classification_criterion_lambda) -- on each iteration classfication_criterion:setTarget(target) must be called
   --local classification_criterion = nn.L1CriterionModule(nn.MSECriterion(), classification_criterion_lambda) -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!

   classification_dictionary.bias:zero()
   if GROUP_SPARISTY_TEN_FIXED_GROUPS then
      classification_dictionary.weight:zero()
      local num_cols = classification_dictionary.weight:size(2)
      for i = 1,10 do
	 classification_dictionary.weight:select(1,i):narrow(1,1+(i-1)*num_cols/10, num_cols/10):fill(1)
      end
   end
   if NORMALIZE_ROWS_OF_CLASS_DICT or BOUND_ROWS_OF_CLASS_DICT then
      classification_dictionary:repair(false, CLASS_DICT_BOUND)
   end

   -- make the criteria_list and filters accessible from the nn.Sequential module for ease of debugging and reporting
   model.criteria_list = criteria_list
   -- each recpool layer generates a separate filter list, which we combine into a common list so we can easily generate collective diagnostics
   model.filter_list = {}
   
   model.layers = layer_list -- used in train_recpool_net to access debug information
   model.disable_pooling = recpool_config_prefs.disable_pooling

   -- take the initial input x and wrap it in a table x [1]
   model:add(nn.IdentityTable()) -- wrap the tensor in a table

   for layer_i = 1,#lambdas do
      -- each layer has an input stage, a feature extraction stage, and a pooling stage.  Since the pooling output of one layer is the input of the next, increment the layer_size index by two per layer
      local current_layer_size = {}
      for i = 1,3 do
	 current_layer_size[i] = layer_size[i + (layer_i-1)*2] 
      end

      -- the global criteria_list is passed into each layer to be built up progressively
      local current_layer = build_recpool_net_layer(layer_i, current_layer_size, lambdas[layer_i], lagrange_multiplier_targets[layer_i], lagrange_multiplier_learning_rate_scaling_factors[layer_i], recpool_config_prefs, criteria_list, data_set, RUN_JACOBIAN_TEST) 
      model:add(current_layer)
      layer_list[layer_i] = current_layer -- this is used in a closure for model:repair()

      -- add this layer's filters to the collective filter lists
      model.filter_list['encoding feature extraction dictionary' .. '_' .. layer_i] = {current_layer.module_list.encoding_feature_extraction_dictionary.weight, 'encoder'}
      model.filter_list['decoding feature extraction dictionary' .. '_' .. layer_i] = {current_layer.module_list.decoding_feature_extraction_dictionary.weight, 'decoder'}
      model.filter_list['explaining away' .. '_' .. layer_i] = {current_layer.module_list.explaining_away.weight, 'encoder'}
      if not(recpool_config_prefs.disable_pooling) then
	 model.filter_list['encoding pooling dictionary' .. '_' .. layer_i] = {current_layer.module_list.encoding_pooling_dictionary.weight, 'encoder'}
	 model.filter_list['decoding pooling dictionary' .. '_' .. layer_i] = {current_layer.module_list.decoding_pooling_dictionary.weight, 'decoder'}
	 model.filter_list['encoding pooling dictionary gradient' .. '_' .. layer_i] = {current_layer.module_list.encoding_pooling_dictionary.gradWeight, 'encoder'}
      end
   end

   -- add the classification dictionary to the collective filter lists
   model.filter_list['classification dictionary'] = {classification_dictionary.weight, 'encoder'}

   local logsoftmax_module = nn.LogSoftMax()
   local safe_sqrt_const = 1e-10
   model:add(nn.SelectTable{1}) -- unwrap the pooled output from the table
   if GROUP_SPARISTY_TEN_FIXED_GROUPS then
      model:add(nn.Square())
   end
   
   -- pass the input through the classification_dictionary, so the set of parameters automatically extracted from the network match those wihtout UNSUPERVISED_CLASSIFICATION, but throw away the output of the classification_dictionary; instead, pass on the input scaled by 0.5*CLASS_DICT_BOUND
   if UNSUPERVISED_CLASSIFICATION then 
      model:add(nn.IdentityTable()) -- wrap the tensor in a table
      local throw_away_class_dict_parallel = nn.ParallelDistributingTable() 
      local class_dict_seq = nn.Sequential()
      class_dict_seq:add(nn.SelectTable{1})
      class_dict_seq:add(classification_dictionary)
      --class_dict_seq:add(logsoftmax_module) -- EXPERIMENTAL CODE!!!
      class_dict_seq:add(nn.ZeroModule())
      class_dict_seq:add(nn.Ignore())
      throw_away_class_dict_parallel:add(class_dict_seq)
      throw_away_class_dict_parallel:add(nn.SelectTable{1})
      model:add(throw_away_class_dict_parallel)
      model:add(nn.SelectTable{2})
      -- compensate for the fact that the rows of the classification_dictionary have L2 norm greater than one, and that softmax is operating over so many more units, all of which are non-negative
      model:add(nn.MulConstant(nil, 0.5*CLASS_DICT_BOUND)) -- with scaling of 1, 0.75 seemed to be a good value; 0.5 is a little too small (categorical units are weak, some part units barely train)
   else
      model:add(classification_dictionary)
   end
   if NORMALIZE_CLASS_DICT_OUTPUT then
      local normalize_classification_dictionary_output = nn.NormalizeTensor()
      model:add(normalize_classification_dictionary_output) 
      --model:add(nn.MulConstant(nil, 5))
   end
   
   if GROUP_SPARISTY_TEN_FIXED_GROUPS then
      model:add(nn.AddConstant(classification_dictionary.weight:size(1), safe_sqrt_const)) -- ensures that we never compute the gradient of sqrt(0), so long as encoding_pooling_dictionary is non-negative with no bias.  Keep in mind that we take the square root of this quantity, so even a seemingly small number like 1e-5 becomes a relatively large constant of 0.00316.  At the same time, the gradient of the square root divides by the square root, and so becomes huge as the argument of the square root approaches zero
      model:add(nn.Sqrt())
      model:add(nn.AddConstant(classification_dictionary.weight:size(1), -math.sqrt(safe_sqrt_const))) -- we add 1e-4 before the square root and subtract (1e-4)^1/2 after the square root, so zero is still mapped to zero, but the gradient contribution is bounded above by 100
   else --if not(UNSUPERVISED_CLASSIFICATION) then -- EXPERIMENTAL CODE!!!
      model:add(logsoftmax_module)
      --model:add(nn.SoftMax()) -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!
   end


   model.module_list = {classification_dictionary = classification_dictionary, 
			logsoftmax = logsoftmax_module}

   if DEBUG_OUTPUT then
      function model:set_target(new_target) print('WARNING: set_target does nothing when using DEBUG_OUTPUT') end
      if IGNORE_CLASSIFICATION then error('cannot use both DEBUG_OUTPUT and IGNORE_CLASSIFICATION simultaneously') end
   else
      if IGNORE_CLASSIFICATION then 
	 model:add(nn.ZeroModule()) -- use if we want to remove the classification criterion for debugging purposes
	 model:add(nn.Ignore())
      else
	 -- make the loss -p*log(p), where p is the softmax of the L2-normalized hidden unit activation; wrap in table, split into self and exponentiated copy, multiply copies 
	 if UNSUPERVISED_CLASSIFICATION then 
	    print('Using UNSUPERVISED CLASSIFICATION!')
	    model:add(nn.IdentityTable()) -- wrap the tensor in a table
	    local entropy_calc_parallel = nn.ParallelDistributingTable() 
	    local recover_prob_seq = nn.Sequential() -- compute prob from the log(prob) returned by the softmax
	    recover_prob_seq:add(nn.SelectTable{1})
	    recover_prob_seq:add(nn.Exp())
	    --recover_prob_seq:add(nn.PrintModule())
	    entropy_calc_parallel:add(recover_prob_seq)
	    entropy_calc_parallel:add(nn.SelectTable{1})
	    model:add(entropy_calc_parallel)
	    model:add(nn.SafeCMulTable()) -- multiply probl and log(prob); returns a Tensor, rather than a table
	    --model:add(nn.PrintModule())
	    --model:add(nn.MulConstant(nil, 20))
	    -- classification_criterion should be an L1_cost, rather than a ClassNLLCriterion
	 end
	 
	 model:add(classification_criterion) 
	 
	 table.insert(criteria_list.criteria, classification_criterion)
	 table.insert(criteria_list.names, 'classification criterion')
	 print('inserting classification negative log likelihood loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      end

      function model:set_target(new_target) 
	 --print('setting classification criterion target to ', new_target)
	 self.current_target = new_target -- FOR DEBUG ONLY!!!
	 classification_criterion:setTarget(new_target) 
      end 

      function model:reset_classification_lambda(new_lambda)
	 classification_criterion:reset_lambda(new_lambda)
      end

      function model:reset_classification_dictionary()
	 classification_dictionary:reset()
	 classification_dictionary.bias:zero()
	 if NORMALIZE_ROWS_OF_CLASS_DICT or BOUND_ROWS_OF_CLASS_DICT then
	    classification_dictionary:repair(false, CLASS_DICT_BOUND)
	 end
      end

      local original_updateOutput = model.updateOutput -- by making this local, it is impossible to access outside of the closures created below
      
      -- note that this is different than the original model.output; model is a nn.Sequential, which ends in the classification_criterion, and thus consists of a single number
      -- it is not desirable to have the model produce a tensor output, since this would imply a corresponding gradOutput, at least in the standard implementation, whereas we want all gradients to be internally generated.  
      function model:updateOutput(input)
	 original_updateOutput(self, input)
	 local summed_loss = 0
	 for i = 1,#(criteria_list.criteria) do
	    if criteria_list.criteria[i].output then -- some criteria may not be defined if we've disabled pooling or other layers
	       summed_loss = summed_loss + criteria_list.criteria[i].output
	    end
	 end
	 -- the output of a tensor is necessary to maintain compatibility with ModifiedJacobian, which directly accesses the output field of the tested module
	 model.output = torch.Tensor(1)
	 model.output[1] = summed_loss
	 return model.output -- while it is tempting to return classification_dictionary.output here, it risks incompatibility with ModifiedJacobian and any other package that expects a single return value
      end

      function model:get_classifier_output()
	 return classification_dictionary.output
      end

      -- model:prepare_train_batch() and model:prepare_test_batch() can be safely called even if dropout (multiplicative_filter) is not used
      if recpool_config_prefs.use_multiplicative_filter then
	 function model:prepare_train_batch() 
	    for i = 1,#layer_list do 
	       layer_list[i]:activate_dropout() 
	       layer_list[i]:randomize_dropout()
	    end 
	 end
	 function model:prepare_test_batch() for i = 1,#layer_list do layer_list[i]:inactivate_dropout() end end
      else
	 function model:prepare_train_batch() end
	 function model:prepare_test_batch() end
      end

      function model:repair()  -- repair each layer; classification dictionary is a normal linear, and so does not need to be repaired
	 for i = 1,#layer_list do
	    layer_list[i]:repair()
	 end
	 if NORMALIZE_ROWS_OF_CLASS_DICT or BOUND_ROWS_OF_CLASS_DICT then
	    classification_dictionary:repair(false, CLASS_DICT_BOUND)
	 end
      end

      function model:reset_ista_learning_scale_factor(new_scale_factor)  -- reset the learning scale factor for each layer; classification dictionary is left unchanged
	 --[[ -- ONLY reset the learning scale factor for the ISTA iterations!!!
	 if NORMALIZE_ROWS_OF_CLASS_DICT then
	    classification_dictionary:reset_learning_scale_factor(new_scale_factor)
	 else
	    error('WARNING: learning scale factor is not changed for classification dictionary!')
	 end
	 --]]
	 
	 for i = 1,#layer_list do
	    layer_list[i]:reset_learning_scale_factor(new_scale_factor)
	 end
      end

      function model:reset_entropy_scale_factor(new_scale_factor)
	 print('Setting entropy scale factor to ' .. new_scale_factor)
	 for i = 1,#layer_list do
	    layer_list[i]:reset_entropy_scale_factor(new_scale_factor)
	 end
      end

      function model:randomize_pooling()
	 for i = 1,#layer_list do
	    layer_list[i]:randomize_pooling()
	 end
      end
   end -- not(DEBUG_OUTPUT)

   --model:add(nn.ParallelDistributingTable('throw away final output')) -- if no modules are added to a ParallelDistributingTable, it throws away its input; updateGradInput produces a tensor of zeros   

   print('criteria_list contains ' .. #(criteria_list.criteria) .. ' entries')      

   return model
end




-- Initialize the parameters to consistent values
local function initialize_parameters(encoding_feature_extraction_dictionary, base_decoding_feature_extraction_dictionary, base_explaining_away, base_shrink, 
				     encoding_pooling_dictionary, decoding_pooling_dictionary, data_set, init_dictionary_min_scaling, init_dictionary_max_scaling,
				     this_layer, layer_size, layer_id, recpool_config_prefs)
   --base_decoding_feature_extraction_dictionary.weight:apply(function(x) return ((x < 0) and 0) or x end) -- make the feature extraction dictionary non-negative, so activities don't need to be balanced in order to get a zero background
   if data_set and (layer_id == 1) then
      for i = 1,base_decoding_feature_extraction_dictionary.weight:size(2) do
	 local selected_column = base_decoding_feature_extraction_dictionary.weight:select(2,i)
	 -- USE 2 FOR STRONG TEMPLATE PRIMING
	 selected_column:add(0.25, data_set.data[math.random(data_set:nExample())]) -- 1 is too much; many units don't train quickly
	 --selected_column:add(-1*selected_column:mean()) -- this is necessary to keep the dynamics stable; otherwise, unit activities tend to oscillate, since the inner product between columns is large, so explaining-away suppression overwhelms the input activation
      end
   end
   base_decoding_feature_extraction_dictionary:repair(true, DEC_FE_DICT_NORM_VALUE) -- make sure that the norm of each column is 1
   encoding_feature_extraction_dictionary.weight:copy(base_decoding_feature_extraction_dictionary.weight:t())

   local selected_enc_cumulative_step_size_init
   if recpool_config_prefs.num_ista_iterations == 1 then
      selected_enc_cumulative_step_size_init = ENC_CUMULATIVE_STEP_SIZE_INIT_1
   elseif recpool_config_prefs.num_ista_iterations >= 10 then
      selected_enc_cumulative_step_size_init = ENC_CUMULATIVE_STEP_SIZE_INIT_10
   else
      error('Do not have initialization specified for ' .. recpool_config_prefs.num_ista_iterations .. ' ISTA iterations')
   end

   if LOCAL_CONNECTIONS_ONLY_EXPLAINING_AWAY_MATRIX then
      base_explaining_away:initialize_local_connection_weight_filter_matrix(LOCAL_CONNECTIONS_ONLY_EXPLAINING_AWAY_MATRIX)
   end
   base_explaining_away.weight:copy(torch.mm(encoding_feature_extraction_dictionary.weight, base_decoding_feature_extraction_dictionary.weight)) -- the step constant should only be applied to explaining_away once, rather than twice; this depends upon the fact that encoding_feature_extraction_dictionary was just set to be the transpose of base_decoding_feature_extraction_dictionary
   if recpool_config_prefs.shrink_style == 'SoftPlus' then
      encoding_feature_extraction_dictionary.weight:mul(math.max(0.1, 10/(recpool_config_prefs.num_ista_iterations + 1)))
   else
      -- scale the encoder so that if all rows are orthogonal and ignoring explaining-away (which have opposite effects), the magnitude of the reconstruction will just match that of the input after num_ista_iterations + 1; the first ista iteration doesn't count towards num_ista_iterations
      encoding_feature_extraction_dictionary.weight:mul(math.min(init_dictionary_max_scaling, math.max(init_dictionary_min_scaling,
												       selected_enc_cumulative_step_size_init/(recpool_config_prefs.num_ista_iterations + 1)))) 
   end
   encoding_feature_extraction_dictionary:repair(FULLY_NORMALIZE_ENC_FE_DICT, 
						 math.min(init_dictionary_max_scaling, math.max(init_dictionary_min_scaling, 
												ENC_CUMULATIVE_STEP_SIZE_BOUND/(recpool_config_prefs.num_ista_iterations + 1))))
   -- multiply by -1, since the explaining_away matrix is added, rather than subtracted
   base_explaining_away.weight:mul(-math.min(init_dictionary_max_scaling, math.max(init_dictionary_min_scaling, selected_enc_cumulative_step_size_init/(recpool_config_prefs.num_ista_iterations + 1)))) 
   --[[
      -- this is only necessary when we preload the feature extraction dictionary with elements of the data set, in which case explaining_away has many strongly negative elements.  
      encoding_feature_extraction_dictionary.weight:mul(1e-1)
      base_explaining_away.weight:mul(0) 
   --]]
   if not(recpool_config_prefs.manually_maintain_explaining_away_diagonal) then -- add the identity matrix into base_explaining_away, assuming this isn't done explicitly by the network dynamics
      for i = 1,base_explaining_away.weight:size(1) do 
	 base_explaining_away.weight[{i,i}] = base_explaining_away.weight[{i,i}] + 1
      end
   end
   base_explaining_away:repair(false, ((BOUND_ELEMENTS_OF_EXPLAINING_AWAY and EXPLAINING_AWAY_BOUND) or 
				       math.max(0.1, EXPLAINING_AWAY_BOUND_SCALE_FACTOR * ENC_CUMULATIVE_STEP_SIZE_BOUND/(recpool_config_prefs.num_ista_iterations + 1))))
   
   if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
      base_shrink.shrink_val:fill(1e-5) --1e-4) --1e-5 -- this should probably be very small, and learn to be the appropriate size!!!
      --base_shrink.shrink_val:fill(1e-2) -- this is only necessary when we preload the feature extraction dictionary with elements from the data set
      base_shrink.negative_shrink_val:mul(base_shrink.shrink_val, -1)
   else
      encoding_feature_extraction_dictionary.bias:fill(-1e-5)
   end
      
   if not(recpool_config_prefs.disable_pooling) then
      if not(recpool_config_prefs.randomize_pooling_dictionary) then
	 decoding_pooling_dictionary.weight:zero() -- initialize to be roughly diagonal
	 for i = 1,layer_size[2] do -- for each unpooled row entry, find the corresponding pooled column entry, plus those beyond it (according to the loop over j)
	    --print('setting entry ' .. i .. ', ' .. math.ceil(i * layer_size[3] / layer_size[2]))
	    for j = 0,0 do
	       decoding_pooling_dictionary.weight[{i, math.min(layer_size[3], j + math.ceil(i * layer_size[3] / layer_size[2]))}] = 1
	    end
	 end
	 
	 --decoding_pooling_dictionary.weight:add(torch.rand(decoding_pooling_dictionary.weight:size()):mul(0.02))
	 --for i = 1,layer_size[3] do
	 --   decoding_pooling_dictionary.weight[{math.random(decoding_pooling_dictionary.weight:size(1)), i}] = 1
	 --end
	 
	 decoding_pooling_dictionary:repair(true) -- make sure that the norm of each column of decoding_pooling_dictionary is 1, even after it is thinned out
	 encoding_pooling_dictionary.weight:copy(decoding_pooling_dictionary.weight:t())
	 --encoding_pooling_dictionary.weight:mul(0.2) -- 1.25 --DEBUG ONLY!!!
	 if recpool_config_prefs.use_squared_weight_matrix then
	    encoding_pooling_dictionary.weight:pow(0.5)
	 end
	 --decoding_pooling_dictionary.weight:add(torch.mul(torch.rand(decoding_pooling_dictionary.weight:size()), 0.3))
	 --encoding_pooling_dictionary.weight:add(torch.mul(torch.rand(encoding_pooling_dictionary.weight:size()), 0.3))
	 encoding_pooling_dictionary:repair(true) -- remember that encoding_pooling_dictionary does not have normalized columns
      end
      
      function this_layer:randomize_pooling(avoid_reset)
	 if not(avoid_reset) then -- when using RUN_JACOBIAN_TEST, constraints are disabled after the first initialization.  A reset here will thus incorrectly include e.g. negative values.
	    decoding_pooling_dictionary:reset()
	 end
	 -- if we don't thin out the pooling dictionary a little, there is no symmetry breaking; all pooling units output about the same output for each input, so the only reliable way to decrease the L1 norm is by turning off all elements.
	 -- on average, initially pool over 40 feature extraction units, regardless of the size of the feature extraction layer
	 local desired_percentage_zeros = 1 - (1 - 0.8) * 200/layer_size[2]
	 decoding_pooling_dictionary:percentage_zeros_per_column(desired_percentage_zeros) --0.8) -- 0.9 works well! -- keep in mind that half of the values are negative, and will be set to zero when repaired
	 --decoding_pooling_dictionary:percentage_zeros_per_column(0.8)
	 decoding_pooling_dictionary:repair(true) -- make sure that the norm of each column of decoding_pooling_dictionary is 1, even after it is thinned out
	 --print('num ones is ' .. decoding_pooling_dictionary.weight:nElement() - torch.eq(decoding_pooling_dictionary.weight, 0):sum())
	 --io.read()
	 
	 encoding_pooling_dictionary.weight:copy(decoding_pooling_dictionary.weight:t())
	 if recpool_config_prefs.use_squared_weight_matrix then
	    encoding_pooling_dictionary.weight:pow(0.5)
	 end
	 --encoding_pooling_dictionary.weight:mul(0.5) -- this helps the initial magnitude of the pooling reconstruction match the actual shrink output -- a value larger than 1 will probably bring the initial values closer to their final values; small columns in the decoding pooling dictionary yield even small reconstructions, since they cause the position units to be small as well
	 encoding_pooling_dictionary:repair(true) -- remember that encoding_pooling_dictionary does not have normalized columns
	 
	 -- DEBUG ONLY
	 --decoding_pooling_dictionary.weight:fill(1)
	 --decoding_pooling_dictionary:repair(true)
      end
      
      if recpool_config_prefs.randomize_pooling_dictionary then
	 this_layer:randomize_pooling(RUN_JACOBIAN_TEST)
      end
      --decoding_pooling_dictionary.weight:zero():add(torch.rand(decoding_pooling_dictionary.weight:size()))
      --decoding_pooling_dictionary:repair(true)
   end -- initialize pooling dictionaries
end



-- input is a table of one element x [1], output is a table of one element s [1]
-- a reconstructing-pooling network.  This is like reconstruction ICA, but with reconstruction applied to both the feature extraction and the pooling, and using shrink operators rather than linear transformations for the feature extraction.  The initial version of this network is built with simple linear transformations, but it can just as easily be used to convolutions
-- use RUN_JACOBIAN_TEST when testing parameter updates
function build_recpool_net_layer(layer_id, layer_size, lambdas, lagrange_multiplier_targets, lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, criteria_list, data_set, RUN_JACOBIAN_TEST) 
   -- lambdas: {ista_L2_reconstruction_lambda, ista_L1_lambda, pooling_L2_shrink_reconstruction_lambda, pooling_L2_orig_reconstruction_lambda, pooling_L2_shrink_position_unit_lambda, pooling_L2_orig_position_unit_lambda, pooling_output_cauchy_lambda, pooling_mask_cauchy_lambda}
   if not(criteria_list) then -- if we're building a multi-layer network, a common criteria list is passed in
      criteria_list = {criteria = {}, names = {}} -- list of all criteria and names comprising the loss function.  These are necessary to run the Jacobian unit test forwards
   end
   RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST or false
   local disable_trainable_pooling = false
   if DEBUG_RF_TRAINING_SCALE == 0 then
      disable_trainable_pooling = true
      print('Disabling trainable pooling!')
      io.read()
   end
   
   if recpool_config_prefs.use_squared_weight_matrix == nil then
      error('recpool_config_prefs.use_squared_weight_matrix was not defined')
   end
   local use_swm = recpool_config_prefs.use_squared_weight_matrix

   -- Build the reconstruction pooling network
   local this_layer = nn.Sequential()

   -- the exact range for the initialization of weight matrices by nn.Linear doesn't matter, since they are rescaled by the normalized_columns constraint
   -- threshold-normalized rows are a bad idea for the encoding feature extraction dictionary, since if a feature is not useful, it will be turned off via the shrinkage, and will be extremely difficult to reactivate later.  It's better to allow the encoding dictionary to be reduced in magnitude.
   local encoding_feature_extraction_dictionary = nn.ConstrainedLinear(layer_size[1],layer_size[2], {no_bias = (recpool_config_prefs.shrink_style == 'ParameterizedShrink'), 
												     normalized_rows = NORMALIZE_ROWS_OF_ENC_FE_DICT}, 
								       RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE, nil, 
								       ((USE_FULL_SCALE_FOR_REPEATED_ISTA_MODULES or RUN_JACOBIAN_TEST) and 1) or 1/(recpool_config_prefs.num_ista_iterations + 1)) 
   local base_decoding_feature_extraction_dictionary = nn.ConstrainedLinear(layer_size[2],layer_size[1], {no_bias = true, normalized_columns = true}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
   local base_explaining_away = nn.ConstrainedLinear(layer_size[2], layer_size[2], 
						     {no_bias = NO_BIAS_ON_EXPLAINING_AWAY, normalized_rows = NORMALIZE_ROWS_OF_EXPLAINING_AWAY, bounded_elements = BOUND_ELEMENTS_OF_EXPLAINING_AWAY,
						     filter_weight_matrix = LOCAL_CONNECTIONS_ONLY_EXPLAINING_AWAY_MATRIX}, 
						     RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE, nil, 
						     ((USE_FULL_SCALE_FOR_REPEATED_ISTA_MODULES or RUN_JACOBIAN_TEST) and 1) or 1/recpool_config_prefs.num_ista_iterations) 
   local base_shrink, shrink_factory
   if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
      base_shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink)
      shrink_factory = function() return duplicate_shrink(base_shrink, layer_size[2], RUN_JACOBIAN_TEST) end
   elseif recpool_config_prefs.shrink_style == 'FixedShrink' then
      shrink_factory = function() return nn.FixedShrink(layer_size[2]) end
      base_shrink = shrink_factory()
   elseif recpool_config_prefs.shrink_style == 'SoftPlus' then
      shrink_factory = function() return nn.SoftPlus() end
      base_shrink = shrink_factory()
   else
      error('shrink style ' .. recpool_config_prefs.shrink_style .. ' was not recognized')
   end

   local decoding_feature_extraction_dictionary_copies = {}
   local explaining_away_copies = {}
   local shrink_copies = {}

   local pooling_dictionary_scaling_factor = 1
   local dpd_training_scale_factor = 1 -- factor by which training of decoding_pooling_dictionary is accelerated
   if not(RUN_JACOBIAN_TEST) then 
      dpd_training_scale_factor = (disable_trainable_pooling and 0) or pooling_dictionary_scaling_factor --5 -- decoding_pooling_dictionary is trained faster than any other module
      --dpd_training_scale_factor = 0 -- DEBUG ONLY!!!
   else -- make sure that all lagrange_multiplier_scaling_factors are -1 when testing, so the update matches the gradient
      for k,v in pairs(lagrange_multiplier_learning_rate_scaling_factors) do
	 if v ~= -1 then
	    error('When doing jacobian test, lagrange_multiplier_scaling_factor ' .. k .. ' was ' .. v .. ' rather than -1')
	 end
      end
   end
   
   if (dpd_training_scale_factor ~= 1) and (DEBUG_shrink or DEBUG_L2 or DEBUG_L1 or DEBUG_OUTPUT) then
      print('SET decoding_pooling_dictionary training scale factor to 1 before testing!!!  Alternatively, turn off debug mode')
      io.read()
   end

   local encoding_pooling_dictionary, decoding_pooling_dictionary 
   if not(recpool_config_prefs.disable_pooling) then
      encoding_pooling_dictionary = nn.ConstrainedLinear(layer_size[2], layer_size[3], 
							 {no_bias = true, non_negative = true, normalized_rows_pooling = NORMALIZE_ROWS_OF_P_FE_DICT, squared_weight_matrix = use_swm}, 
							 RUN_JACOBIAN_TEST, (disable_trainable_pooling and 0) or pooling_dictionary_scaling_factor) -- this should have zero bias
      
      decoding_pooling_dictionary = nn.ConstrainedLinear(layer_size[3], layer_size[2], {no_bias = true, normalized_columns_pooling = true, non_negative = true, squared_weight_matrix = false}, RUN_JACOBIAN_TEST, dpd_training_scale_factor) -- this should have zero bias, and columns normalized to unit magnitude
   end

   -- there's really no reason to define these here rather than where they're used
   local use_lagrange_multiplier_for_L1_regularizer = false

   -- the entirety of feature_extraction_sparsifying_module is used as the criterion, so it's safe to modify it using standard nn modules without disrupting learning or unit testing, so long as it properly returns a scalar.  The input is a tensor, rather than a table
   local pooling_sparsifying_module, mask_sparsifying_module
   local function feature_extraction_sparsifying_module_factory(lambda_scale_factor, disable_pure_L1)
      local pure_L1_switch = ((disable_pure_L1 and 0) or 1)
      if use_lagrange_multiplier_for_L1_regularizer then
	 --return nn.ParameterizedL1Cost(layer_size[2], lambda_scale_factor * lambdas.ista_L1_lambda, lagrange_multiplier_targets.feature_extraction_target, lagrange_multiplier_learning_rate_scaling_factors.feature_extraction_scaling_factor, RUN_JACOBIAN_TEST) 
	 return nn.L1CriterionModule(nn.L1Cost(), lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda) 
      elseif USE_HETEROGENEOUS_L1_SCALING_FACTOR then -- Use two different values for L1 on disjoint subsets of the hidden units
	 return build_heterogeneous_L1_criterion(layer_size, 0.5 * lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda, 
						 lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda) 
      elseif USE_L1_OVER_L2_NORM then 
	 -- first is pure L1, second is L1 over L2
	 return build_L1_over_L2_criterion(0.75 * lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda, 
					   0.25 * lambda_scale_factor * lambdas.ista_L1_lambda) 
      elseif USE_PROB_WEIGHTED_L1 then 
	 -- -2 seems to be too strong; -1 may be about right
	 return build_weighted_L1_criterion(WEIGHTED_L1_ENTROPY_SCALING * lambda_scale_factor, 
					    WEIGHTED_L1_PURE_L1_SCALING * lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda)
      elseif USE_HARD_ENTROPY then
	 return build_hard_entropy_criterion(WEIGHTED_L1_ENTROPY_SCALING * lambda_scale_factor, 
					    WEIGHTED_L1_PURE_L1_SCALING * lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda)
      else
	 return nn.L1CriterionModule(nn.L1Cost(), lambda_scale_factor * pure_L1_switch * lambdas.ista_L1_lambda) -- NORMAL VERSION
	 
	 -- EXPERIMENTAL VERSION - use either \sum_i (1 - |z_i| / (\sum_j |z_j|)) * |z_i| if the last argument < 1; or \sum_i (1 - a*|z_i|/(\sum_j |z_j|))^n * |z_i|, where a is the last argument, if the last argument is >= 1
	 --feature_extraction_sparsifying_module_factory = function() return nn.L1CriterionModule(nn.L1Cost(), lambdas.ista_L1_lambda, nil, nil, 1) end -- EXPERIMENTAL VERSION 
      end
   end

   
   if use_lagrange_multiplier_for_L1_regularizer then
      --pooling_sparsifying_module = nn.ParameterizedL1Cost(layer_size[3], lambdas.pooling_output_cauchy_lambda, lagrange_multiplier_targets.pooling_target, lagrange_multiplier_learning_rate_scaling_factors.pooling_scaling_factor, RUN_JACOBIAN_TEST) 
      pooling_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_output_cauchy_lambda) 

      mask_sparsifying_module = nn.ParameterizedL1Cost(layer_size[2], lambdas.pooling_mask_cauchy_lambda, lagrange_multiplier_targets.mask_target, lagrange_multiplier_learning_rate_scaling_factors.mask_scaling_factor, RUN_JACOBIAN_TEST) 
      --mask_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_mask_cauchy_lambda) 
   else
      -- the L1CriterionModule, rather than the wrapped criterion, produces the correct scaled error      
      pooling_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_output_cauchy_lambda) 
      --local pooling_sparsifying_loss_function = nn.L1Cost() --nn.CauchyCost(lambdas.pooling_output_cauchy_lambda) -- remove lambda from build function if we ever switch back to a cauchy cost!
      mask_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_mask_cauchy_lambda) 
   end

   local copied_decoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary_transpose, decoding_feature_extraction_dictionary_transpose_straightened_copy = 
      duplicate_decoding_feature_extraction_dictionary(base_decoding_feature_extraction_dictionary, {layer_size[1], layer_size[2]}, true, RUN_JACOBIAN_TEST) -- next-to-last arg is make_transpose
   
   -- it's easier to create a single module list, which can be indexed with pairs, than to add each separately to the nn.Sequential module corresponding to this layer
   -- there's a separate debug_module_list defined below that contains the internal nn.Sequential modules, so we can see the processing flow explicitly
   local module_list = {encoding_feature_extraction_dictionary = encoding_feature_extraction_dictionary, 
			decoding_feature_extraction_dictionary = base_decoding_feature_extraction_dictionary,
			copied_decoding_feature_extraction_dictionary = copied_decoding_feature_extraction_dictionary,
			decoding_feature_extraction_dictionary_transpose = decoding_feature_extraction_dictionary_transpose,
			decoding_feature_extraction_dictionary_transpose_straightened_copy = decoding_feature_extraction_dictionary_transpose_straightened_copy,
			explaining_away = base_explaining_away, 
			shrink = base_shrink, 
			explaining_away_copies = explaining_away_copies,
			shrink_copies = shrink_copies,
			encoding_pooling_dictionary = encoding_pooling_dictionary, 
			decoding_pooling_dictionary = decoding_pooling_dictionary, 
			--feature_extraction_sparsifying_module = feature_extraction_sparsifying_module, -- now that we're potentially creating many of these, just save the first after it is made
			pooling_sparsifying_module = pooling_sparsifying_module, 
			mask_sparsifying_module = mask_sparsifying_module}
   local entropy_crit_list = {}

   -- Initialize the parameters to consistent values
   local init_dictionary_min_scaling = 0.01 -- USE 0 FOR STRONG TEMPLATE PRIMING
   local init_dictionary_max_scaling = 1
   initialize_parameters(encoding_feature_extraction_dictionary, base_decoding_feature_extraction_dictionary, base_explaining_away, base_shrink, 
			 encoding_pooling_dictionary, decoding_pooling_dictionary, data_set, init_dictionary_min_scaling, init_dictionary_max_scaling, 
			 this_layer, layer_size, layer_id, recpool_config_prefs)


   -- take the input x [1] and calculate the sparse code z [1], the transformed input W*x [2], and the untransformed input x [3]
   local ista_seq
   --ista_seq, shrink_copies[#shrink_copies + 1] = build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink)
   ista_seq = build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink) -- this function does not actually copy shrink, so a shrink_copy does not need to be returned
   this_layer:add(ista_seq)

   local base_multiplicative_filter
   if recpool_config_prefs.use_multiplicative_filter then
      base_multiplicative_filter = nn.MultiplicativeFilter(layer_size[2], false)
      local multiplicative_filter_parallel = build_multiplicative_filter_layer(base_multiplicative_filter, false) -- last argument indicates whether this is the last ISTA iteration, in which case we should also pass on the offset shrink
      this_layer:add(multiplicative_filter_parallel)

      function this_layer:activate_dropout() base_multiplicative_filter:activate() end
      function this_layer:inactivate_dropout() base_multiplicative_filter:inactivate() end
      function this_layer:randomize_dropout() base_multiplicative_filter:randomize() end
   end

   if recpool_config_prefs.num_ista_iterations < recpool_config_prefs.num_loss_function_ista_iterations then
      error('not presently configured to handle num_ista_iterations = ' .. recpool_config_prefs.num_ista_iterations .. ' > num_loss_function_ista_iteration = ' .. 
	    recpool_config_prefs.num_loss_function_ista_iterations)
   end

   for i = 1,recpool_config_prefs.num_ista_iterations do
      --calculate the sparse code z [1]; preserve the transformed input W*x [2], and the untransformed input x [3]
      local use_base_explaining_away = (i == 1) -- the base_explaining_away must be used once directly for parameter flattening in nn.Module to work properly; base_shrink is already used by build_ISTA_first_iteration
      local this_explaining_away = duplicate_linear_explaining_away(base_explaining_away, {layer_size[1], layer_size[2]}, recpool_config_prefs.num_ista_iterations, use_base_explaining_away, 
								    RUN_JACOBIAN_TEST)
      local this_shrink = shrink_factory()

      local this_offset_shrink
      if CREATE_BUFFER_ON_L1_LOSS and (i == recpool_config_prefs.num_ista_iterations) then
	 this_offset_shrink = nn.FixedShrink(layer_size[2])
	 module_list.offset_shrink = module_list.offset_shrink or this_offset_shrink
	 if recpool_config_prefs.shrink_style ~= 'FixedShrink' then
	    error('offset shrink is only compatible with fixed shrink')
	 end
      end

      explaining_away_copies[#explaining_away_copies + 1], shrink_copies[#shrink_copies + 1] = this_explaining_away, this_shrink
      ista_seq = build_ISTA_iteration(this_explaining_away, this_shrink, RUN_JACOBIAN_TEST, (i == recpool_config_prefs.num_ista_iterations), this_offset_shrink, recpool_config_prefs.manually_maintain_explaining_away_diagonal) -- last argument indicates whether this is the last ISTA iteration, in which case we should also pass on the offset shrink
      this_layer:add(ista_seq)

      if recpool_config_prefs.use_multiplicative_filter and base_multiplicative_filter then 
	 local this_multiplicative_filter = duplicate_multiplicative_filter(base_multiplicative_filter, layer_size[2])
	 local multiplicative_filter_parallel = build_multiplicative_filter_layer(this_multiplicative_filter, (i == recpool_config_prefs.num_ista_iterations)) -- last argument indicates whether this is the last ISTA iteration, in which case we should also pass on the offset shrink
	 this_layer:add(multiplicative_filter_parallel)
      end

      if recpool_config_prefs.num_ista_iterations - i + 1 <= recpool_config_prefs.num_loss_function_ista_iterations then
	 print('adding copy ' .. recpool_config_prefs.num_ista_iterations - i + 1 .. ' of the L2 reconstruction, L1 sparsity, and entropy loss functions')
	 -- add the reconstruction, sparsifying, and entropy loss.  Ideally, by enforcing these losses over multiple iterations, the network will be induced to saturate at good values, rather than having categorical-units increase monotonically throughout the trajectory.  Make sure the scale down losses proprortional to the number of times they are repeated.  Consider repeating the classification loss as well, although this seems both more costly computationally and more cumbersome, since I've generally thrown away the previous hidden state in the process of calculating these terms.
	 

	 local current_decoding_fe_dict = base_decoding_feature_extraction_dictionary
	 if #decoding_feature_extraction_dictionary_copies ~= 0 then -- don't duplicate for the first copy
	    current_decoding_fe_dict = duplicate_decoding_feature_extraction_dictionary(base_decoding_feature_extraction_dictionary, {layer_size[1], layer_size[2]}, false, RUN_JACOBIAN_TEST) -- next-to-last arg is make_transpose
	 end
	 table.insert(decoding_feature_extraction_dictionary_copies, current_decoding_fe_dict)
	    
	 -- reconstruct the input D*z [4] from the code z [1], leaving z [1], the transformed input W*x [2], and the original input x [3] unchanged; offset shrink goes from [4] to [5] if it is present
	 this_layer:add(linearly_reconstruct_input(current_decoding_fe_dict)) --base_decoding_feature_extraction_dictionary))
	 -- calculate the L2 dist between the reconstruction based on the shrunk code D*z [4], and the original input x [3]; discard all signals but the layer n code z [1], the transformed input W*x [2], the untransformed layer n-1 input x[3], and possibly the layer n-1 offset shrink output [4], as well as the L2 loss criterion
	 -- crit must be distinct for each copy
	 local L2_reconstruction_loss_module, L2_reconstruction_criterion = build_L2_reconstruction_loss(L2_RECONSTRUCTION_SCALING_FACTOR * lambdas.ista_L2_reconstruction_lambda / 
													 recpool_config_prefs.num_loss_function_ista_iterations, criteria_list) 
	 module_list.L2_reconstruction_criterion = module_list.L2_reconstruction_criterion or L2_reconstruction_criterion -- save the first copy
	 this_layer:add(L2_reconstruction_loss_module) 
	 
	 -- input and output are the subject of the shrink operation z [1], the transformed input W*x [2], the original input x [3], the possibly the offset shrink output [4]
	 local feature_extraction_sparsifying_module = feature_extraction_sparsifying_module_factory(1 / recpool_config_prefs.num_loss_function_ista_iterations) -- criteria must be distinct for each copy, since it is saved in the criteria_list for jacobian testing
	 local ista_sparsifying_loss_seq = build_sparsifying_loss(feature_extraction_sparsifying_module, criteria_list, CREATE_BUFFER_ON_L1_LOSS)
	 module_list.feature_extraction_sparsifying_module = module_list.feature_extraction_sparsifying_module or feature_extraction_sparsifying_module -- save the first copy
	 if feature_extraction_sparsifying_module.reset_entropy_scale_factor then -- only add to entropy_crit_list if there is an entropy_scale_factor to reset
	    table.insert(entropy_crit_list, feature_extraction_sparsifying_module)
	 end
	 this_layer:add(ista_sparsifying_loss_seq)
	 
	 -- input and output are the subject of the shrink operation z [1], the transformed input W*x [2], the original input x [3]
	 if LOW_PASS_FILTERED_ENTROPY then
	    print('WARNING: using low-pass filtered entropy!  This does not seem to reduce the number of categorical-units!')
	    io.read()

	    local low_pass_sparsifying_module = feature_extraction_sparsifying_module_factory(100 * 1 / recpool_config_prefs.num_loss_function_ista_iterations, true) -- criteria must be distinct for each copy, since it is saved in the criteria_list for jacobian testing; last argument disables L1 component of the loss function
	    local low_pass_seq = nn.Sequential()
	    low_pass_seq:add(nn.LowPass(0.01, RUN_JACOBIAN_TEST))
	    low_pass_seq:add(low_pass_sparsifying_module)
	    local low_pass_sparsifying_loss = build_sparsifying_loss(low_pass_seq, criteria_list, false)
	    if low_pass_sparsifying_module.reset_entropy_scale_factor then -- only add to entropy_crit_list if there is an entropy_scale_factor to reset
	       table.insert(entropy_crit_list, low_pass_sparsifying_module)
	    end
	    this_layer:add(low_pass_sparsifying_loss)
	 end	    

	 -- input and output are the subject of the shrink operation z [1], the transformed input W*x [2], the original input x [3]
	 if ELASTIC_NET_LOSS then
	    local L2_elastic_net_loss = build_sparsifying_loss(nn.L1CriterionModule(nn.L2Cost(ELASTIC_NET_LOSS * lambdas.ista_L1_lambda / recpool_config_prefs.num_loss_function_ista_iterations, 1,
											      false)), 
							       criteria_list, false)
	    this_layer:add(L2_elastic_net_loss)
	 end
	 
      end
   end

   -- throw away all terms but the sparse code z [1] and the original input x [2], since that's what's expected by the pooling layers
   local discard_ista_terms = nn.ParallelDistributingTable()
   discard_ista_terms:add(nn.SelectTable{1})
   discard_ista_terms:add(nn.SelectTable{3})
   this_layer:add(discard_ista_terms)
   

   -- pool the input z [1] to obtain the pooled code s = sqrt(Q*z^2) [1], the preserved sparse code z [2], and the original input x[3]
   local pooling_seq, pooling_L2_loss_seq, pooling_L2_debug_module_list, pooling_sparsifying_loss_seq
   if not(recpool_config_prefs.disable_pooling) then
      pooling_seq = build_pooling(encoding_pooling_dictionary)
      this_layer:add(pooling_seq)

      --[[
	 -- the gradient of the L2 position loss is only negative if P*s >= sqrt(alpha)
	 -- add a constant additive offset to the pooling units
	 local offset_pt = nn.ParallelTable()
	 local lambda_ratio = math.min((lambdas.pooling_L2_shrink_position_unit_lambda + lambdas.pooling_L2_orig_position_unit_lambda) / (lambdas.pooling_L2_shrink_reconstruction_lambda + lambdas.pooling_L2_orig_reconstruction_lambda), 1e5)
	 offset_pt:add(nn.AddConstant(layer_size[3], math.sqrt(lambda_ratio))) -- / (layer_size[2] * layer_size[3]))))
	 offset_pt:add(nn.Identity())
	 offset_pt:add(nn.Identity())
	 this_layer:add(offset_pt)
      --]]
      
      --[[
   -- divisively normalize the pooling units before reconstruction
	 local normalize_pt = nn.ParallelTable()
	 local normalize_seq = nn.Sequential()
	 --local normalize_pooled_output = nn.NormalizeTensor()
	 --normalize_seq:add(normalize_pooled_output)
	 normalize_seq:add(nn.MulConstant(layer_size[3], 1)) --50*math.sqrt(2)))
	 normalize_pt:add(normalize_seq)
	 normalize_pt:add(nn.Identity())
	 normalize_pt:add(nn.Identity())
   this_layer:add(normalize_pt)
      --]]
      
      -- calculate the L2 reconstruction and position loss for pooling; return the pooled code s [1] and the original input x [2]
      pooling_L2_loss_seq, pooling_L2_debug_module_list =
	 build_pooling_L2_loss(decoding_pooling_dictionary, copied_decoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary_transpose, mask_sparsifying_module, lambdas, criteria_list)
      this_layer:add(pooling_L2_loss_seq)
      
      -- calculate the L1 magnitude of the pooling code s [1] (also contains the original input x[2]), returning the pooling code s [1] and the original input x [2] unchanged
      pooling_sparsifying_loss_seq = build_sparsifying_loss(pooling_sparsifying_module, criteria_list)
      this_layer:add(pooling_sparsifying_loss_seq)
   end -- use pooling

   local extract_pooled_output = nn.ParallelDistributingTable('remove unneeded original input x') -- ensure that the original input x [2] receives a gradOutput of zero
   extract_pooled_output:add(nn.SelectTable{1})
   this_layer:add(extract_pooled_output)

   -- THIS SHOULD PROBABLY ONLY BE DONE BEFORE THE CLASSIFICATION LOSS!!!
   -- normalize the output of each layer; output is normalized s [1]
   local normalize_output = nn.NormalizeTable()
   this_layer:add(normalize_output) -- this is undefined if all outputs are zero -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!


   this_layer.module_list = module_list
   -- used when checking for nans in train_recpool_net
   this_layer.debug_module_list = pooling_L2_debug_module_list or {}
   this_layer.debug_module_list.ista_sparsifying_loss_seq = ista_sparsifying_loss_seq
   this_layer.debug_module_list.pooling_seq = pooling_seq
   this_layer.debug_module_list.pooling_sparsifying_loss_seq = pooling_sparsifying_loss_seq
   this_layer.debug_module_list.normalize_output = normalize_output
   this_layer.debug_module_list.normalize_pooled_output = normalize_pooled_output

   -- the constraints on the parameters of these modules cannot be enforced by updateParameters when used in conjunction with the optim package, since it adjusts the parameters directly
   -- THIS IS A HACK!!!
   local repair_counter = 0
   --function this_layer:repair()
   --end
   
   function this_layer:repair()
      -- normalizing these two large dictionaries is the slowest part of the algorithm, consuming perhaps 75% of the running time.  Normalizing less often obviously increases running speed considerably.  We'll need to evaluate whether it's safe...
      if repair_counter == 0 then
	 encoding_feature_extraction_dictionary:repair(FULLY_NORMALIZE_ENC_FE_DICT, math.min(init_dictionary_max_scaling, math.max(init_dictionary_min_scaling, ENC_CUMULATIVE_STEP_SIZE_BOUND/(recpool_config_prefs.num_ista_iterations + 1))))
	 base_decoding_feature_extraction_dictionary:repair(FULLY_NORMALIZE_DEC_FE_DICT, DEC_FE_DICT_NORM_VALUE) -- force full normalization of columns
      end
      repair_counter = (repair_counter + 1) % recpool_config_prefs.repair_interval
      
      base_explaining_away:repair(false, ((BOUND_ELEMENTS_OF_EXPLAINING_AWAY and EXPLAINING_AWAY_BOUND) or 
					  math.max(0.1, EXPLAINING_AWAY_BOUND_SCALE_FACTOR * ENC_CUMULATIVE_STEP_SIZE_BOUND/(recpool_config_prefs.num_ista_iterations + 1))))
      if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
	 base_shrink:repair() -- repairing the base_shrink doesn't help if the parameters aren't linked!!!
      end
      if not(recpool_config_prefs.disable_pooling) and not(disable_trainable_pooling) then
	 encoding_pooling_dictionary:repair() --true) -- force full normalization of rows -- DEBUG ONLY!!!  FOR THE LOVE OF GOD!!  RECONSIDER THIS!!!
	 decoding_pooling_dictionary:repair(true) -- force full normalization of columns
      end
   end

   function this_layer:reset_learning_scale_factor(new_scale_factor)
      encoding_feature_extraction_dictionary:reset_learning_scale_factor(new_scale_factor)
      --base_decoding_feature_extraction_dictionary:reset_learning_scale_factor(new_scale_factor) -- covered by the iteration over decoding_feature_extraction_dictionary_copies below
      copied_decoding_feature_extraction_dictionary:reset_learning_scale_factor(new_scale_factor)
      decoding_feature_extraction_dictionary_transpose:reset_learning_scale_factor(new_scale_factor)
      for i = 1,#decoding_feature_extraction_dictionary_copies do
	 decoding_feature_extraction_dictionary_copies[i]:reset_learning_scale_factor(new_scale_factor)
      end
      base_explaining_away:reset_learning_scale_factor(new_scale_factor)
      for i = 1,#explaining_away_copies do
	 explaining_away_copies[i]:reset_learning_scale_factor(new_scale_factor)
      end
      if not(recpool_config_prefs.disable_pooling) and not(disable_trainable_pooling) then
	 encoding_pooling_dictionary:reset_learning_scale_factor(new_scale_factor)
	 decoding_pooling_dictionary:reset_learning_scale_factor(new_scale_factor)
      end
   end

   function this_layer:reset_entropy_scale_factor(new_entropy_scale_factor)
      for _,crit in ipairs(entropy_crit_list) do
	 crit:reset_entropy_scale_factor(new_entropy_scale_factor)
      end
   end


   --return this_layer, criteria_list, encoding_feature_extraction_dictionary, base_decoding_feature_extraction_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, feature_extraction_sparsifying_module, pooling_sparsifying_module, mask_sparsifying_module, base_explaining_away, base_shrink, explaining_away_copies, shrink_copies
   return this_layer
end

-- Test full reconstructing-pooling model with Jacobian.  Maintain a list of all Criteria.  When running forward, sum over all Criteria to determine the gradient of a single unified energy (should be able to just run updateOutput on the Criteria).  When running backward, just call updateGradInput on the network.  No gradOutput needs to be provided, since all modules terminate in an output-free Criterion.
