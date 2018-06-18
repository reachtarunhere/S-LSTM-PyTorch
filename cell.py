import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenGate(nn.Module):

    def __init__(self, hidden_size, input_size, bias, nonlinearity="sigmoid"):
        super(HiddenGate, self).__init__()
        self.linear = nn.Linear(
            3*hidden_size + input_size + hidden_size, hidden_size, bias=bias)
        self.nonlinearity = F.sigmoid if nonlinearity == "sigmoid" else F.tanh

    def forward(self, Xis, x_i, prev_g):
        return self.nonlinearity(self.linear(torch.cat([Xis, x_i, prev_g])))


class SentenceStateGate(nn.Module):

    def __init__(self, hidden_size, input_size, bias):
        super(SentenceStateGate, self).__init__()
        self.linear = nn.Linear(
            hidden_size + hidden_size, hidden_size, bias=bias)

    def forward(self, prev_g, h):
        """ h is either h_av or h_i for different i"""
        return F.sigmoid(self.linear(torch.cat([prev_g, h])))


class SLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(SLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # hidden state gates

        self.i_i_op = HiddenGate(hidden_size, input_size, bias)
        self.l_i_op = HiddenGate(hidden_size, input_size, bias)
        self.r_i_op = HiddenGate(hidden_size, input_size, bias)
        self.f_i_op = HiddenGate(hidden_size, input_size, bias)
        self.s_i_op = HiddenGate(hidden_size, input_size, bias)
        self.o_i_op = HiddenGate(hidden_size, input_size, bias)

        self.u_i_op = HiddenGate(hidden_size, input_size, nonlinearity="tanh")

        # sentence state gates

        self.g_f_g_op = SentenceStateGate(hidden_size, input_size, bias)
        self.g_f_i_op = SentenceStateGate(hidden_size, input_size, bias)
        self.g_o_op = SentenceStateGate(hidden_size, input_size, bias)

    def reset_params(self):
        pass

    def get_Xis(self, prev_h_states):
        """Apply proper index selection mask to get xis"""
        # How do you handle it getting shorter eh??
        pass

    def forward(self, prev_h_states, prev_c_states, prev_g_state,
                x_i, prev_c_g):

        Xi_i = self.get_Xi_i(prev_h_states)

        i_i = self.i_i_op(Xi_i, x_i, prev_g_state)
        l_i = self.l_i_op(Xi_i, x_i, prev_g_state)
        r_i = self.l_i_op(Xi_i, x_i, prev_g_state)
        f_i = self.l_i_op(Xi_i, x_i, prev_g_state)
        s_i = self.l_i_op(Xi_i, x_i, prev_g_state)
        o_i = self.l_i_op(Xi_i, x_i, prev_g_state)

        u_i = self.u_i_op(Xi_i, x_i, prev_g_state)

        # Now Get Softmaxed Versions

        i_i, l_i, r_i, f_i, s_i = self.softmaxed_gates(
            [i_i, l_i, r_i, f_i, s_i])

        # what happens to the the last cell here?????? which has no i+1?
        # what happens when first one has no i-1??

        prev_c_left, prev_c_right, prev_c = self.get_prev_cs(prev_c_states)

        c_i = l_i * prev_c_left + f_i * prev_c + r_i * prev_c_right + \
            s_i * prev_c_g + i_i * u_i

        h_i = o_i * F.tanh(c_i)
        # Now for the sentence level calculations

        h_avg = prev_h_states.mean(dim=0)

        g_f_g = self.g_f_g_op(prev_g_state, h_avg)
        g_f_i = self.g_f_i_op(prev_g_state, prev_h_states)
        g_o = self.g_o_op(prev_g_state, h_avg)

        temp = self.softmaxed_gates(list(torch.unbind(g_f_i)) + [g_f_g])

        g_f_i = torch.stack(temp[:-1], dim=0)
        g_f_g = temp[-1]

        c_g = g_f_g * prev_c_g + torch.sum(g_f_i * prev_c_states, dim=0)

        g = g_o * F.tanh(c_g)

        return h_i, c_i, g, c_g

    def softmaxed_gates(self, gates_list):
        softmaxed = F.softmax(torch.stack(gates_list), dim=0)
        return torch.unbind(softmaxed)
