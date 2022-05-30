from deepmusic.moduleloader import ModuleLoader
from deepmusic.keyboardcell import KeyboardCell
# encapsulate sing data so we can run get_scale, get_relative methods
import deepmusic.songstruct as music
import numpy as np
import tensorflow as tf


def build_network(self):
    # create computation graph, encapsulate  session and the graph init (basic tensorflow stuff)
    input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()

    # note data
    with tf.name_scope('placeholder_inputs'):
        self.inputs = [
            tf.placeholder(
                tf.float32,
                [self.args.batch_size, input_dim],
                name='inputs'  # how much data
            )
        ]

        # targets (whether or not one of the 88 keys on a piano was pressed) -> binary classification probem

    with tf.name_scope('placeholder_targets'):
        self.targets = [
            tf.placeholder(
                tf.int32,  # (either 0 or 1)
                [self.batch_size],
                name='targets'
            )
        ]

    with tf.name_scope('placeholder_use_prev'):
        self.use_prev = [
            tf.placeholder(
                tf.bool,
                [],
                name='use_prev'
            )
        ]

    # defining the network
    # loop function to connect one of the outputs of one network to the next input of another network
    self.loop_processing = ModuleLoader.loop_processing.build_module(
        self.inputs, self.targets, self.use_prev)

    def loop_rnn(prev, i):
        next_input = self.loop_processing(prev)
        return tf.cond(self.prev[i], lambda: next_input, lambda: self.inputs[i])

    # building the sequence of sequences model
    self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(
        decoder_inputs=self.inputs,  # defined in keyboard cell
        initial_state=None,
        cell=KeyboardCell,
        loop_function=loop_rnn
    )
    # training
    # defining a loss function
    # because this is multi class classification, cross entropy is the loss function
    # it measures the difference between two/more probability distributions

    loss_fct = tf.nn.seq2seq.sequence_loss(
        self.outputs,
        self.targets,
        softmax_loss_function=tf.nn.softmax.cross_entropy_with_logits
    )

    # we have a sequence of notes, looking at each note predict what
    # the next note will be. we want to minimise the loss function
    # by looking at the difference between predicted actual next note

    # the notes are represented by floats
