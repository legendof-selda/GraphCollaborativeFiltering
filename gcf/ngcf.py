from .base import BaseCF


class NGCF(BaseCF):
    """
    This implementation of collbarative filtering is based on
    https://dl.acm.org/doi/10.1145/3331184.3331267
    https://github.com/xiangwang1223/neural_graph_collaborative_filtering
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_cat,
        n_fold=100,
        emb_propogation_layers=[64],
        node_dropout=False,
        **kwargs,
    ):
        self.emb_propogation_layers = emb_propogation_layers
        self.n_propogating_layers = len(self.emb_propogation_layers)
        self.node_dropout = node_dropout

        super().__init__(n_users, n_items, n_cat, **kwargs)

    def create_variables(self):
        super().create_variables()

        self.propogation_layers_weights = {}

        # appending the embedding size as first shape of the propogation layer
        user_emb_prop = [self.user_emb_size] + self.emb_propogation_layers
        item_emb_prop = [self.item_emb_size] + self.emb_propogation_layers

        for layer in range(self.n_propogating_layers):
            name = f"weight_user_propogated_L{layer}"
            self.propogation_layers_weights[name] = tf.Variable(
                self.initializer(user_emb_prop[layer], user_emb_prop[layer + 1]),
                name=name,
            )

            name = f"bias_user_propogated_L{layer}"
            self.propogation_layers_weights[name] = tf.Variable(
                self.initializer(1, user_emb_prop[layer + 1]), name=name
            )

            name = f"weight_item_propogated_L{layer}"
            self.propogation_layers_weights[name] = tf.Variable(
                self.initializer(item_emb_prop[layer], item_emb_prop[layer + 1]),
                name=name,
            )

            name = f"bias_item_propogated_L{layer}"
            self.propogation_layers_weights[name] = tf.Variable(
                self.initializer(1, item_emb_prop[layer + 1]), name=name
            )
