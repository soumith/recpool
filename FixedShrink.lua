-- A module that performs a one-sided "soft"-shrink operation, where the magnitude of the shrink is fixed at zero; that is, it takes the maximum of the input and zero

local FixedShrink, parent = torch.class('nn.FixedShrink', 'nn.Module')

-- EFFICIENCY NOTE: when using non-negative units this could be accomplished more efficiently using an unparameterized, one-sided rectification, just like Glorot, Bordes, and Bengio, along with a non-positive bias in the inverse_dictionary.  However, both nn.SoftShrink and the shrinkage utility method implemented in kex are two-sided.

function FixedShrink:__init(size)
   parent.__init(self)
   --self.output:resize(size)
   self.gradInput:resize(size)

   --[[
   self.max_output = torch.Tensor(size,1)
   self.selected_index = torch.LongTensor(size)

   self.input_and_threshold = torch.Tensor(size, 2):zero()
   self.just_input = self.input_and_threshold:select(2,1)
   --]]
end


function FixedShrink:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input)
   self.output[torch.lt(input, 0)] = 0
   -- self.output:maxZero()
   return self.output
end

function FixedShrink:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   -- this assumes that updateOutput was called before updateGradInput
   self.gradInput[torch.lt(input, 0)] = 0 -- The second index always holds 0, so if it is selected by max, the output was clipped
   -- self.gradInput:maxZero2(input)
   
   return self.gradInput
end


-- Koray's shrinkage implementation in kex is two-sided, whereas we want a one-sided operation, to parallel soft_plus.
--[[
function FixedShrink:updateOutputCareful(input)
   if input:dim() ~= 1 then
      error('FixedShrink expects one-dimensional inputs')
   end
   
   if self.input_and_threshold:size(1) ~= input:size(1) then
      self.input_and_threshold:resize(input:size(1), 2)
      self.input_and_threshold:zero()

      self.just_input = self.input_and_threshold:select(2,1)

      self.max_output:resize(input:size(1), 1)
      self.selected_index:resize(input:size(1))
      error('resizing FixedShrink')
   end
      
   self.just_input:copy(input)
   torch.max(self.max_output, self.selected_index, self.input_and_threshold, 2)

   self.output = self.max_output:select(2,1) 

   --print(self.max_output:size())
   --print(self.output:size())

   return self.output
end

-- take this from nonsmooth gradient calculation
function FixedShrink:updateGradInputCareful(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   -- this assumes that updateOutput was called before updateGradInput
   self.gradInput[torch.eq(self.selected_index, 2)] = 0 -- The second index always holds 0, so if it is selected by max, the output was clipped

   return self.gradInput
   end
--]]

function FixedShrink:repair()
   -- does nothing, since FixedShrink is not parameterized
end
