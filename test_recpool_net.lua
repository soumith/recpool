require 'torch'
require 'nn'
dofile('init.lua')
dofile('build_recpool_net.lua')

PARAMETER_UPDATE_METHOD = 'optim' -- 'module' -- when updating using the optim package, each set of shared parameters is updated only once and gradients must be shared; when updating using the module methods of the nn package, every instance of a set of shared parameters is updated individually, and gradients should not be shared.  Make sure this matches the setting in build_recpool_net.lua

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local rec_pool_test = {}
local run_test = {}
local other_tests = {}

function rec_pool_test.Square()
   local in1 = torch.rand(10,20)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Square()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.Square1D()
   local in1 = torch.rand(10)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.Square()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.Sqrt()
   local in1 = torch.rand(10,20)
   local module = nn.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = jac.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input, 0, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

--[[
function rec_pool_test.DebugSquareSquare()
   local in1 = torch.rand(10,20)
   local module = nn.DebugSquare('square')
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   --local inj = math.random(5,10)
   --local ink = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.DebugSquare('square')

   local err = jac.testJacobian(module, input, -2, 2)
   mytester:assertlt(err, precision, 'error on state ')

   --local ferr, berr = jac.testIO(module, input, -2, 2)
   --mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.DebugSquareSqrt()
   local in1 = torch.rand(10,20)
   local module = nn.DebugSquare('sqrt')
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   --local inj = math.random(5,10)
   --local ink = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.DebugSquare('sqrt')

   local err = jac.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   --local ferr, berr = jac.testIO(module, input, 0.1, 2)
   --mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end
--]]


function rec_pool_test.ConstrainedLinearLinearWeightMatrix()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink,ini):zero()
   local module = nn.ConstrainedLinear(ini, inj, {no_bias = true, normalized_columns = true, squared_weight_matrix = false}, true, 1, true) 

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.ConstrainedLinearSquaredWeightMatrix()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink,ini):zero()
   local module = nn.ConstrainedLinear(ini, inj, {no_bias = true, normalized_columns = true, squared_weight_matrix = true}, true, 1, true) 

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end



function rec_pool_test.LogSoftMax()
   local ini = math.random(10,20)
   --local inj = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.LogSoftMax()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, expprecision, 'error on state ') -- THIS REQUIRES LESS PRECISION THAN NORMAL, presumably because the exponential tends to make backpropagation unstable

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.FixedShrink()
   local ini = math.random(10,20)
   --local inj = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.FixedShrink(ini)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ') 

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end



function rec_pool_test.AddConstant()
   local input = torch.rand(10,20)
   local random_addend = math.random()
   local module = nn.AddConstant(input:size(), random_addend)
   local out = module:forward(input)
   local err = out:dist(input:add(random_addend))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.AddConstant(input:size(), random_addend)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.MulConstant()
   local input = torch.rand(10,20)
   local random_addend = math.random()
   local module = nn.MulConstant(input:size(), random_addend)
   local out = module:forward(input)
   local err = out:dist(input:mul(random_addend))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.MulConstant(input:size(), random_addend)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.AppendConstant()
   local input = torch.rand(10,20)
   local random_append = math.random()
   local module = nn.AppendConstant(random_append)
   local out = module:forward(input)
   local err_orig = out:narrow(2, 1, input:size(2)):dist(input)
   local err_append = out:narrow(2, input:size(2)+1, 1):dist(torch.Tensor(input:size(1)):fill(random_append))
   mytester:asserteq(err_orig, 0, torch.typename(module) .. ' - forward err orig ')
   mytester:asserteq(err_append, 0, torch.typename(module) .. ' - forward err append ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.AppendConstant(random_append, 3)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.LowPass()
   local random_mixing_factor = math.random()
   local module = nn.LowPass(random_mixing_factor)
   local num_rows = 5
   local num_cols = 10
   local input, accum
   
   for i = 1,20 do
      --print('iter ' .. i)
      input = torch.rand(num_rows,num_cols)
      
      if i == 1 then -- initialize accum to something sensible for the first iteration
	 accum = input:sum(1):div(input:size(1))
      end

      local expanded_accum = torch.mul(accum, 1-random_mixing_factor):expandAs(input) 
      local predicted_output = torch.add(expanded_accum, random_mixing_factor, input)

      accum:mul(1-random_mixing_factor)
      --print('TEST before accum update ', accum)
      --print('TEST adding ', torch.mul(input:sum(1), random_mixing_factor/num_rows))
      accum:add(random_mixing_factor/num_rows, input:sum(1))

      --print('TEST end iter accum ', accum)

      local out = module:forward(input)
      local err = out:dist(predicted_output)
      --print('diff is ' .. err)
      mytester:assertlt(err, 1e-10, torch.typename(module) .. ' - forward err ')
      --io.read()
   end


   -- one dimension
   local ini = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.LowPass(random_mixing_factor, true)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state (1D) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (1D) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (1D) ')

   -- two dimensions
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local input = torch.Tensor(inj, ini):zero()

   local module = nn.LowPass(random_mixing_factor, true)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state (2D) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (2D) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (2D) ')
end


function rec_pool_test.MultiplicativeFilter()
   local input = torch.rand(10,20)
   local module = nn.MultiplicativeFilter(input:size(2))
   local out = module:forward(input)
   local err = 0
   for i = 1,input:size(1) do
      err = err + out:select(1,i):dist(torch.cmul(input:select(1,i), module.bias_filter))
   end
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local input = torch.Tensor(inj, ini):zero()

   local module = nn.MultiplicativeFilter(input:size(2))

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end





function create_parameterized_shrink_test(require_nonnegative_units)
   local function this_parameterized_shrink_test()
      local size = math.random(10,20)
      local module = nn.ParameterizedShrink(size, require_nonnegative_units, true) -- first, try with units that can be negative; ignore nonnegativity constraint on shrink values
      local shrink_vals = torch.rand(size)
      module:reset(shrink_vals)
      local input = torch.Tensor(size):zero()
      
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')
      local err = jac.testJacobianParameters(module, input, module.shrink_val, module.grad_shrink_val)
      mytester:assertlt(err,precision, 'error on shrink val ')
   
      local err = jac.testJacobianUpdateParameters(module, input, module.shrink_val)
      mytester:assertlt(err,precision, 'error on shrink val [direct update]')
      
      for t,err in pairs(jac.testAllUpdate(module, input, 'shrink_val', 'grad_shrink_val')) do
	 mytester:assertlt(err, precision, string.format(
			      'error on shrink val (testAllUpdate) [%s]', t))
      end
      
      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   end

   return this_parameterized_shrink_test
end

rec_pool_test.ParameterizedShrinkNonnegative = create_parameterized_shrink_test(true)
rec_pool_test.ParameterizedShrinkUnconstrained = create_parameterized_shrink_test(false)

function rec_pool_test.ParameterizedL1Cost()
   local size = math.random(10,20)
   local initial_lambda = math.random()
   local desired_criterion_value = math.random()
   local learning_rate_scaling_factor = -1 -- necessary since normally the error is maximized with respect to the lagrange multipliers
   local module = nn.ParameterizedL1Cost(size, initial_lambda, desired_criterion_value, learning_rate_scaling_factor, 'full test')
   
   local input = torch.Tensor(size):zero()
   
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')
   
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update]')
   
   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format('error on weight (testAllUpdate) [%s]', t))
   end
   
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CAddTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CAddTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CosineDistance()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CosineDistance()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.PairwiseDistance()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.PairwiseDistance(2)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CMulTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CMulTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CDivTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CDivTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SafePower()
   local in1 = torch.rand(10,20)
   local module = nn.SafePower(2)
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local pw = torch.uniform()*math.random(1,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SafePower(pw)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.SafeLog()
   local in1 = torch.rand(10,20)
   local offset = 1e-5
   local module = nn.SafeLog(offset)
   local out = module:forward(in1)
   local err = out:dist(in1:add(offset):log())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SafeLog()

   local err = nn.Jacobian.testJacobian(module, input, 1e-5, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input, 1e-5, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SafeEntropy()
   local in1 = torch.rand(10,20)
   local offset = 1e-5
   local module = nn.SafeEntropy(offset)
   local out = module:forward(in1)
   local log_calc = in1:clone():add(offset):log()
   local err = out:dist(log_calc:cmul(in1):mul(-1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SafeEntropy()

   local err = nn.Jacobian.testJacobian(module, input, 0, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input, 0, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SafeCMulTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.SafeCMulTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.NormalizeTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(inj,ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.NormalizeTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.NormalizeTensor1D()
   local ini = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.NormalizeTensor()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.NormalizeTensor2D()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(inj,ini):zero()
   local module = nn.NormalizeTensor()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end



function rec_pool_test.NormalizeTensorL11D()
   local ini = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.NormalizeTensorL1()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.NormalizeTensorL12D()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(inj,ini):zero()
   local module = nn.NormalizeTensorL1()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SumWithinBatch()
   local ini = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.SumWithinBatch()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1D) ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (1D, non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (1D) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (1D) ')

   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(inj,ini):zero()
   local module = nn.SumWithinBatch()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2D) ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (2D, non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (2D) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (2D) ')
end


function rec_pool_test.L2Cost()
   print(' testing L2Cost!!!')
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.L2Cost(math.random(), 2)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2 inputs) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.L2Cost(math.random(), 1)

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1 input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err ')


   input = torch.Tensor(ini):zero()
   module = nn.L2Cost(math.random(), 1, true) -- test true L2 cost, the sqrt-of-sum-of-squares, rather than just the sum-of-squares

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1 input, sqrt-of-sum-of-squares) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err ')
end

function rec_pool_test.L1OverL2Cost()
   print(' testing L1OverL2Cost!!!')
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini, inj):zero()
   local module = nn.L1OverL2Cost()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2d input) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2d input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2d input) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.L1OverL2Cost()

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1d input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1d input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1d input) - i/o backward err ')
end


function rec_pool_test.SumCriterionModule()
   print(' testing SumCriterionModule!!!')
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini, inj):zero()
   local module = nn.SumCriterionModule()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2d input) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2d input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2d input) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.SumCriterionModule()

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1d input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1d input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1d input) - i/o backward err ')
end





function rec_pool_test.SoftClassNLLCriterion()
   local aggregate_incorrect_prob_options = {false, true}
   for crit_options = 1,#aggregate_incorrect_prob_options do
      local ini = math.random(10,20)
      local inj = math.random(10,20)
      local input = torch.Tensor(ini, inj):zero()
      local module = nn.L1CriterionModule(nn.SoftClassNLLCriterion(aggregate_incorrect_prob_options[crit_options]), 1)
      local target = torch.Tensor(ini)
      for i = 1,ini do
	 target[i] = math.random(input:size(2))
      end
      module:setTarget(target)
      
      local err = jac.testJacobianTable(module,input)
      mytester:assertlt(err,precision, 'error on state (2 inputs) with option ' .. crit_options)
      
      local ferr,berr = jac.testIOTable(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err with option ' .. crit_options)
      mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err with option ' .. crit_options)
      
      input = torch.Tensor(ini):zero()
      module = nn.L1CriterionModule(nn.SoftClassNLLCriterion(aggregate_incorrect_prob_options[crit_options]), 1)
      module:setTarget(math.random(ini))
      
      err = jac.testJacobianTable(module,input)
      mytester:assertlt(err,precision, 'error on state (1 input) with option ' .. crit_options)
      
      ferr,berr = jac.testIOTable(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err with options ' .. crit_options)
      mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err with options ' .. crit_options)
   end
end

function rec_pool_test.HingeClassNLLCriterion()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(ini, inj):zero()
   local module = nn.L1CriterionModule(nn.HingeClassNLLCriterion(), 1)
   local target = torch.Tensor(ini)
   for i = 1,ini do
      target[i] = math.random(input:size(2))
   end
   module:setTarget(target)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2 inputs) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.L1CriterionModule(nn.HingeClassNLLCriterion(), 1)
   module:setTarget(math.random(ini))

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1 input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err ')
end


function rec_pool_test.CauchyCost()
   print(' testing CauchyCost!!!')
   local ini = math.random(10,20)
   --local inj = math.random(10,20)
   --local ink = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.CauchyCost(math.random())

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')
end


function rec_pool_test.ParallelIdentity()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.ParallelTable()
   module:add(nn.Identity())
   module:add(nn.Identity())

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CopyTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local table_size = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   -- test CopyTable on a tensor (non-table) input
   local module = nn.Sequential()
   module:add(nn.IdentityTable())
   module:add(nn.CopyTable(1, math.random(1,5)))

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   
   -- test CopyTable on an a table input with many entries
   input = {}
   for i=1,table_size do
      input[i] = torch.Tensor(ini):zero()
   end
   module = nn.CopyTable(math.random(1,table_size), math.random(1,5))

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function rec_pool_test.IdentityTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local table_size = math.random(5,10)
   local input = torch.Tensor(ini):zero()
   local module = nn.IdentityTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SelectTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local table_size = math.random(5,10)
   local input = {}
   for i=1,table_size do
      input[i] = torch.Tensor(ini):zero()
   end

   -- try using SelectTable to pass through the inputs unchanged
   local module = nn.ParallelDistributingTable()
   for i=1,table_size do
      module:add(nn.SelectTable{i})
   end
   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- try using SelectTable to permute the inputs
   module = nn.ParallelDistributingTable()
   for i=1,table_size-1 do
      module:add(nn.SelectTable{i+1})
   end
   module:add(nn.SelectTable{1})
   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')


   -- try using SelectTable to throw away all but one input
   module = nn.ParallelDistributingTable()
   module:add(nn.SelectTable{math.random(table_size)})
   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')


   -- try using SelectTable to feed into CAddTables
   module = nn.ParallelDistributingTable()
   local current_module
   for i=1,table_size-1 do
      current_module = nn.Sequential()
      current_module:add(nn.SelectTable{i, i+1})
      current_module:add(nn.CAddTable())
      module:add(current_module)
   end

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
end


function rec_pool_test.L1CriterionModule()
   local ini = math.random(10,20)
   local lambda = math.random()
   local input = torch.Tensor(ini):zero()
   local module = nn.L1CriterionModule(nn.L1Cost(), lambda)

   print('L1CriterionModule test')

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


-- dparam must be a table off all gradients that correspond to shared copies of the param being tested
local function test_module(name, module, module_copies, param, grad_param, parameter_list, model, input, jac, precision, min_param, max_param)
   local function copy_table(t)
      local t2 = {}
      for k,v in pairs(t) do
	 t2[k] = v
      end
      return t2
   end
   local unused_params

   print(name .. ' ' .. param)
   local grads = {module[grad_param]}
   if module_copies then
      for i,v in ipairs(module_copies) do
	 table.insert(grads, v[grad_param])
      end
   end
   local err = jac.testJacobianParameters(model, input, module[param], grads, min_param, max_param)
   mytester:assertlt(err,precision, 'error on ' .. name .. ' ' .. param)
   unused_params = copy_table(parameter_list)
   for k,v in pairs(parameter_list) do
      if module[param] == v then
	 --print('removing key ' .. k .. ' for ' .. name .. '.' .. param)
	 table.remove(unused_params, k)
      end
   end
   --table.remove(unused_params, param_number)
   local err = jac.testJacobianUpdateParameters(model, input, module[param], unused_params, min_param, max_param)
   mytester:assertlt(err,precision, 'error on ' .. name .. ' ' .. param .. ' [full processing chain, direct update] ')
end


function rec_pool_test.full_network_test()
   --REMEMBER that all Jacobian tests randomly reset the parameters of the module being tested, and then return them to their original value after the test is completed.  If gradients explode for only one module, it is likely that this random initialization is incorrect.  In particular, the signals passing through the explaining_away matrix will explode if it has eigenvalues with magnitude greater than one.  The acceptable scale of the random initialization will decrease as the explaining_away matrix increases, so be careful when changing layer_size.

   -- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer, repair_interval
   local recpool_config_prefs = {}
   recpool_config_prefs.num_ista_iterations = 10
   recpool_config_prefs.num_loss_function_ista_iterations = 5
   --recpool_config_prefs.shrink_style = 'ParameterizedShrink'
   recpool_config_prefs.shrink_style = 'FixedShrink' --'ParameterizedShrink'
   --recpool_config_prefs.shrink_style = 'SoftPlus'
   recpool_config_prefs.disable_pooling = false
   if recpool_config_prefs.disable_pooling then
      print('POOLING IS DISABLED!!!')
      io.read()
   end
   recpool_config_prefs.use_squared_weight_matrix = true
   recpool_config_prefs.normalize_each_layer = false
   recpool_config_prefs.repair_interval = 1
   recpool_config_prefs.manually_maintain_explaining_away_diagonal = true
   recpool_config_prefs.use_multiplicative_filter = true -- do dropout with nn.MultiplicativeFilter?

   --local layer_size = {math.random(10,20), math.random(10,20), math.random(5,10), math.random(5,10)} 
   --local layer_size = {math.random(10,20), math.random(10,20), math.random(5,10), math.random(10,20), math.random(5,10), math.random(5,10)} 
   --local layer_size = {math.random(10,20), math.random(10,20), math.random(5,10), math.random(10,20), math.random(5,10), math.random(10,20), math.random(5,10), math.random(5,10)} 
   local minibatch_size = 0
   local layer_size = {10, 25, 10, 10}
   local target
   if minibatch_size > 0 then
      target = torch.Tensor(minibatch_size)
      for i = 1,minibatch_size do
	 target[i] = math.random(layer_size[#layer_size])
      end
   else
      target = math.random(layer_size[#layer_size])
   end
   --local target = torch.zeros(layer_size[#layer_size]) -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!
   --target[math.random(layer_size[#layer_size])] = 1

   local lambdas = {ista_L2_reconstruction_lambda = math.random(), 
		    ista_L1_lambda = math.random(), 
		    pooling_L2_shrink_reconstruction_lambda = math.random(), 
		    pooling_L2_orig_reconstruction_lambda = math.random(), 
		    pooling_L2_shrink_position_unit_lambda = math.random(), 
		    pooling_L2_orig_position_unit_lambda = math.random(), 
		    pooling_output_cauchy_lambda = math.random(), 
		    pooling_mask_cauchy_lambda = math.random()}

   local lagrange_multiplier_targets = {feature_extraction_target = math.random(), pooling_target = math.random(), mask_target = math.random()}
   local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = -1, pooling_scaling_factor = -1, mask_scaling_factor = -1}

   ---[[
   local layered_lambdas = {lambdas} --{lambdas, lambdas}
   local layered_lagrange_multiplier_targets = {lagrange_multiplier_targets} --{lagrange_multiplier_targets, lagrange_multiplier_targets}
   local layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors} --{lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors}
   --]]

   --[[
   local layered_lambdas = {lambdas, lambdas, lambdas}
   local layered_lagrange_multiplier_targets = {lagrange_multiplier_targets, lagrange_multiplier_targets, lagrange_multiplier_targets}
   local layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors}
   --]]
   
   local model =
      build_recpool_net(layer_size, layered_lambdas, 1, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, nil, true) -- final true -> NORMALIZATION IS DISABLED!!!
   print('finished building recpool net')

   -- create a list of all the parameters of all modules, so they can be held constant when doing Jacobian tests
   local parameter_list = {}
   for i = 1,#model.layers do
      for k,v in pairs(model.layers[i].module_list) do
	 if v.parameters and v:parameters() then -- if a parameters function is defined
	    local params = v:parameters()
	    for j = 1,#params do
	       table.insert(parameter_list, params[j])
	    end
	 end
      end
   end
   
   for k,v in pairs(model.module_list) do
      if v.parameters and v:parameters() then -- if a parameters function is defined
	 local params = v:parameters()
	 for j = 1,#params do
	    table.insert(parameter_list, params[j])
	 end
      end
   end

   model:set_target(target)
   
   print('Since the model contains a LogSoftMax, use precision ' .. expprecision .. ' rather than ' .. precision)
   local precision = 3*expprecision
   
   local input
   if minibatch_size > 0 then
      input = torch.Tensor(minibatch_size, layer_size[1]):zero()
   else
      input = torch.Tensor(layer_size[1]):zero()
   end

   -- check that we don't always produce nans
   local check_for_non_nans = false
   if check_for_non_nans then
      local test_input = torch.rand(layer_size[1])
      model:updateOutput(test_input)
      print('check that we do not always produce nans')
      print(test_input)
      print(model.module_list.classification_dictionary.output)
      io.read()
   end


   local err = jac.testJacobian(model, input)
   mytester:assertlt(err,precision, 'error on processing chain state ')

   for i = 1,#model.layers do
      local dec_fe_dict_duplicates
      if PARAMETER_UPDATE_METHOD == 'module' then
	 dec_fe_dict_duplicates = {model.layers[i].module_list.copied_decoding_feature_extraction_dictionary, 
				   model.layers[i].module_list.decoding_feature_extraction_dictionary_transpose_straightened_copy}
      end
      test_module('decoding dictionary weight', model.layers[i].module_list.decoding_feature_extraction_dictionary, dec_fe_dict_duplicates, 'weight', 'gradWeight', 
		  parameter_list, model, input, jac, precision)
      -- the failure of the test on the bias is OK, since the bias is normally set equal to zero.  In fact, the gradient on the bias isn't really defined, since the transposed copy has a different dimensionality than the other instances of decoding_feature_extraction_dictionary
      test_module('decoding dictionary bias', model.layers[i].module_list.decoding_feature_extraction_dictionary, dec_fe_dict_duplicates, 'bias', 'gradBias', 
		  parameter_list, model, input, jac, precision)

      -- problem here!!!
      test_module('encoding dictionary weight', model.layers[i].module_list.encoding_feature_extraction_dictionary, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      test_module('encoding dictionary bias', model.layers[i].module_list.encoding_feature_extraction_dictionary, {}, 'bias', 'gradBias', parameter_list, model, input, jac, precision)

      local exp_away_duplicates = {} -- the first element of explaining_away_copies is identical to the base explaining_away; explaining_away_copies includes *all* copies
      if PARAMETER_UPDATE_METHOD == 'module' then
	 for j = 2,#(model.layers[i].module_list.explaining_away_copies) do
	    table.insert(exp_away_duplicates, model.layers[i].module_list.explaining_away_copies[j])
	 end
      end
      -- don't allow large weights, or the messages exhibit exponential growth
      test_module('explaining away weight', model.layers[i].module_list.explaining_away, exp_away_duplicates, 'weight', 'gradWeight', 
		  parameter_list, model, input, jac, precision, -0.6, 0.6)
      test_module('explaining away bias', model.layers[i].module_list.explaining_away, exp_away_duplicates, 'bias', 'gradBias', 
		  parameter_list, model, input, jac, precision)
      if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
	 local shrink_duplicates
	 if PARAMETER_UPDATE_METHOD == 'module' then
	    shrink_duplicates = model.layers[i].module_list.shrink_copies
	 end
	 test_module('shrink shrink_val', model.layers[i].module_list.shrink, shrink_duplicates, 'shrink_val', 'grad_shrink_val', 
		     parameter_list, model, input, jac, precision, 2e-4, 0.01) -- it doesn't make sense for the shrink parameter to be negative
      end
      -- element 8 of the parameter_list is negative_shrink_val

      if not(disable_pooling) then
	 test_module('decoding pooling dictionary weight', model.layers[i].module_list.decoding_pooling_dictionary, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision, 0, 2)
	 test_module('decoding pooling dictionary bias', model.layers[i].module_list.decoding_pooling_dictionary, {}, 'bias', 'gradBias', parameter_list, model, input, jac, precision, 0, 2)
	 
	 -- make sure that the random weights assigned to the encoding pooling dictionary for Jacobian testing are non-negative!
	 test_module('encoding pooling dictionary weight', model.layers[i].module_list.encoding_pooling_dictionary, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision, 0, 2)
	 test_module('encoding pooling dictionary bias', model.layers[i].module_list.encoding_pooling_dictionary, {}, 'bias', 'gradBias', parameter_list, model, input, jac, precision, 0, 2)
      end      

      if model.layers[i].module_list.feature_extraction_sparsifying_module.weight then
	 test_module('feature extraction sparsifying module', model.layers[i].module_list.feature_extraction_sparsifying_module, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      end
      if model.layers[i].module_list.pooling_sparsifying_module.weight then
	 test_module('pooling sparsifying module', model.layers[i].module_list.pooling_sparsifying_module, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      end
      if model.layers[i].module_list.mask_sparsifying_module.weight then
	 test_module('mask sparsifying module', model.layers[i].module_list.mask_sparsifying_module, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      end
   end
   
   test_module('classification dictionary weight', model.module_list.classification_dictionary, {}, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
   test_module('classification dictionary bias', model.module_list.classification_dictionary, {}, 'bias', 'gradBias', parameter_list, model, input, jac, precision)


end

function rec_pool_test.ISTA_reconstruction()
   -- check that ISTA actually finds a sparse reconstruction.  decoding_dictionary.output should be similar to test_input, and shrink_copies[#shrink_copies].output should have some zeros

   -- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer, repair_interval
   local recpool_config_prefs = {}
   recpool_config_prefs.num_ista_iterations = 50
   recpool_config_prefs.num_loss_function_ista_iterations = 10
   --recpool_config_prefs.shrink_style = 'ParameterizedShrink'
   recpool_config_prefs.shrink_style = 'FixedShrink' --'ParameterizedShrink'
   --recpool_config_prefs.shrink_style = 'SoftPlus'
   recpool_config_prefs.disable_pooling = false
   recpool_config_prefs.use_squared_weight_matrix = true
   recpool_config_prefs.normalize_each_layer = false
   recpool_config_prefs.repair_interval = 1
   recpool_config_prefs.manually_maintain_explaining_away_diagonal = true


   local minibatch_size = 0
   local layer_size = {10, 64, 10, 10}
   local target
   if minibatch_size > 0 then
      target = torch.Tensor(minibatch_size)
      for i = 1,minibatch_size do
	 target[i] = math.random(layer_size[#layer_size])
      end
   else
      target = math.random(layer_size[#layer_size])
   end


   local lambdas = {ista_L2_reconstruction_lambda = math.random(), 
		    ista_L1_lambda = math.random(), 
		    pooling_L2_shrink_reconstruction_lambda = math.random(), 
		    pooling_L2_orig_reconstruction_lambda = math.random(), 
		    pooling_L2_shrink_position_unit_lambda = math.random(), 
		    pooling_L2_orig_position_unit_lambda = math.random(), 
		    pooling_output_cauchy_lambda = math.random(), 
		    pooling_mask_cauchy_lambda = math.random()}
   local lagrange_multiplier_targets = {feature_extraction_target = math.random(), pooling_target = math.random(), mask_target = math.random()}
   local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = -1, pooling_scaling_factor = -1, mask_scaling_factor = -1}

   local layered_lambdas = {lambdas}
   local layered_lagrange_multiplier_targets = {lagrange_multiplier_targets}
   local layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors}

   -- create the dataset so the features of the network can be initialized
   local data = nil
   --require 'mnist'
   --local data = mnist.loadTrainSet(500, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  

   --Indexing labels returns an index, rather than a tensor
   --data:normalizeL2() -- normalize each example to have L2 norm equal to 1


   local model =
      build_recpool_net(layer_size, layered_lambdas, 1, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, nil) -- final true -> NORMALIZATION IS DISABLED!!!

   -- convenience names for easy access
   local shrink_copies = model.layers[1].module_list.shrink_copies
   local shrink = model.layers[1].module_list.shrink
   local explaining_away_copies = model.layers[1].module_list.explaining_away_copies
   local explaining_away = model.layers[1].module_list.explaining_away
   local decoding_feature_extraction_dictionary = model.layers[1].module_list.decoding_feature_extraction_dictionary

   local test_input 
   if minibatch_size > 0 then
      test_input = torch.rand(minibatch_size, layer_size[1]) --torch.Tensor(minibatch_size, layer_size[1]):zero()
   else
      test_input = torch.rand(layer_size[1])
   end

   model:set_target(target)

   model:updateOutput(test_input)
   print('test input', test_input)
   print('reconstructed input', model.layers[1].module_list.decoding_feature_extraction_dictionary.output)
   print('shrink output', shrink_copies[#shrink_copies].output)

   local test_gradInput = torch.zeros(model.output:size())
   model:updateGradInput(test_input, test_gradInput)


   if shrink_style == 'ParameterizedShrink' then
      -- confirm that parameter sharing is working properly
      for i = 1,#shrink_copies do
	 if (shrink_copies[i].shrink_val:storage() ~= shrink.shrink_val:storage()) or (shrink_copies[i].grad_shrink_val:storage() ~= shrink.grad_shrink_val:storage()) then
	    print('ERROR!!!  shrink_copies[' .. i .. '] does not share parameters with base shrink!!!')
	    io.read()
	 end
	 --print('shrink_copies[' .. i .. '] gradInput', shrink_copies[i].gradInput)
	 --print('shrink_copies[' .. i .. '] output', shrink_copies[i].output)
      end
   end
   
   for i = 1,#explaining_away_copies do
      if (explaining_away_copies[i].weight:storage() ~= explaining_away.weight:storage()) or (explaining_away_copies[i].bias:storage() ~= explaining_away.bias:storage()) then
	 print('ERROR!!!  explaining_away_copies[' .. i .. '] does not share parameters with base explaining_away!!!')
	 io.read()
      end
      --print('explaining_away_copies[' .. i .. '] gradInput', explaining_away_copies[i].gradInput)
      --print('explaining_away_copies[' .. i .. '] output', explaining_away_copies[i].output)
   end


   ---[[
   if minibatch_size == 0 then
      local shrink_output_tensor = torch.Tensor(decoding_feature_extraction_dictionary.output:size(1), #shrink_copies)
      for i = 1,#shrink_copies do
	 shrink_output_tensor:select(2,i):copy(decoding_feature_extraction_dictionary:updateOutput(shrink_copies[i].output))
      end
      print(shrink_output_tensor)
   else
      local index_list = {1, 2, 3, 4, 5, 6, 7}
      local num_shrink_output_tensor_elements = #index_list -- model.layers[1].module_list.shrink.output:size(1)
      local shrink_output_tensor = torch.Tensor(num_shrink_output_tensor_elements, 1 + #model.layers[1].module_list.shrink_copies)
      
      for j = 1,#index_list do
	 shrink_output_tensor[{j, 1}] = model.layers[1].module_list.shrink.output[{1, index_list[j]}] -- minibatch_size >= 1, so we need to select the minibatch from which to draw the state
      end
      
      for i = 1,#model.layers[1].module_list.shrink_copies do
	 --shrink_output_tensor:select(2,i):copy(model.layers[1].module_list.shrink_copies[i].output)
	 for j = 1,#index_list do
	    shrink_output_tensor[{j, i+1}] = model.layers[1].module_list.shrink_copies[i].output[{1, index_list[j]}] -- minibatch_size >= 1, so we need to select the minibatch from which to draw the state
	 end
      end

      print('full evolution of index_list for the first element of the minibatch', shrink_output_tensor)
   end
   
   --]]

end







--local num_tests = 0 
--for name in pairs(rec_pool_test) do num_tests = num_tests + 1; print('test ' .. num_tests .. ' is ' .. name) end
--print('number of tests: ', num_tests)
math.randomseed(os.clock())

mytester:add(rec_pool_test)
--mytester:add(run_test)
--mytester:add(other_tests)

jac = nn.Jacobian
mytester:run()

