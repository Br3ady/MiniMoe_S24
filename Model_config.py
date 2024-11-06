class Config(object):
    def __init__(
            self,
            vocab_size=50257,
            n_positions=2048,
            n_ctx=2048,
            n_embd=2048,
            n_layer=6,
            n_head=16,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,

    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range