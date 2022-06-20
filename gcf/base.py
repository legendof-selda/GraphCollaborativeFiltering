import tensorflow as tf

tfph = tf.compat.v1.placeholder


class BaseCF(tf.Model):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cat: int,
        user_emb_size=32,
        item_emb_size=32,
        cat_emb_size=4,
        batch_size=32,
        hyperparams={},
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_cat = n_cat
        self.user_emb_size = user_emb_size
        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.batch_size = batch_size

        self.set_hyperparams(hyperparams)

        self.create_variables()
        self.init_optimizer()

    @property
    def default_hyperparams(self) -> dict:
        """Returns the default hyperparameters needed by model.

        Returns:
            dict: Default hyperparameters
        """
        return {
            "emb_l2_reg_lambda": 1e-5,
            "category_emb_l2_lambda": 1e-5,
            "rmse_lambda": 1e-5,
            "lr": 0.01,
        }

    def set_hyperparams(self, hyperparams: dict):
        """Sets the hyperparamets as attributes in the model

        Args:
            hyperparams (dict): The user given hyperparams which will override the default_hyperparams
        """
        hyperparams = {**self.default_hyperparams, **hyperparams}
        for hyperparam, value in hyperparams.items():
            setattr(self, hyperparam, value)

    def define_inputs(self):
        self.cat_adj = tfph(
            dtype=tf.int32,
            shape=(self.n_cat, self.n_cat),
            name="category_adjacency_matrix",
        )

    def create_variables(self):
        """Initializes the required variables"""
        self.initializer = tf.initializers.glorot_uniform()
        # Create embeddings
        self.user_embedding = tf.Variable(
            self.initializer((self.n_users + 1, self.user_emb_size)),
            name="user_embedding",
        )
        self.item_embedding = tf.Variable(
            self.initializer((self.n_items + 1, self.item_emb_size)),
            name="item_embedding",
        )
        self.cat_embedding = tf.Variable(
            self.initializer((self.n_cat, self.cat_emb_size)), name="cat_embedding"
        )

    @property
    def cat_tree(self) -> tf.Tensor:
        """Multiplies the category embedding with the input category adjacency
        to generate the heirarchial representation of category

        Returns:
            tf.Tensor: The cat tree
        """
        # output = n_cat * cat_emb_size
        return self.cat_adj * self.cat_embedding

    def bpr_loss(self, users, positive_items, negative_items):
        """Bayesian Personalization Ranking (BPF) loss.
        Computes the BPF loss.
        https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf

        Args:
            users (_type_): The user embeddings
            positive_items (_type_): The positive items embeddings. (Positive items are items which the user has interacted with.)
            negative_items (_type_): The negative items embeddings. (Negative items are items which the user hasn't interacted with.)
        """
        pos_scores = tf.reduce_sum(tf.multiply(users, positive_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, negative_items), axis=1)

        return -1 * tf.reduce_mean(tf.log(tf.nn.sigmoid(pos_scores - neg_scores)))

    def embedding_regularization_loss(self, users, positive_items, negative_items):
        """Calculate L2 loss for the embeddings.

        Args:
            users (_type_): The user embeddings
            positive_items (_type_): The positive items embeddings. (Positive items are items which the user has interacted with.)
            negative_items (_type_): The negative items embeddings. (Negative items are items which the user hasn't interacted with.)
        """
        l2_loss = tf.nn.l2_loss
        reg_loss = l2_loss(users) + l2_loss(positive_items) + l2_loss(negative_items)
        reg_loss = reg_loss / self.batch_size * self.emb_l2_reg_lambda
        return reg_loss

    def category_emb_l2_loss(self):
        """Calculate L2 loss for category tree embedding.

        Returns:
            tf.Tensor: L2 Loss
        """
        return (
            tf.nn.l2_loss(self.cat_tree) / self.batch_size * self.category_emb_l2_lambda
        )

    def rmse_loss(self, user_items_truth, user_items_pred, mask):
        """Calculates RMSE loss for the predicted matrix factorization

        Args:
            user_items_truth (_type_): User item matrix (ground truth)
            user_items_pred (_type_): User item matrix (predicted)
            mask (_type_): Mask denoting, user_item relationship that is found in dataset.
        """
        return (
            tf.sqrt(tf.reduce_mean((user_items_truth - user_items_pred) * mask))
            * self.rmse_lambda
        )

    def loss(self):
        """Compute total loss used for optimizer minimize function"""
        raise NotImplementedError("Create loss function for optimizer")

    def init_optimizer(self):
        """Initialize Optimizer"""
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.opt_minimize = self.optimizer.minimize(self.loss, tf.trainable_variables())
