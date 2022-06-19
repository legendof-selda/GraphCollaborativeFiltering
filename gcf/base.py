import tensorflow as tf


class BaseCF(tf.Model):
    def __init__(self, n_users: int, n_items: int, n_cat: int, user_emb_size=32, item_emb_size=32, batch_size=32, hyperparams={}):
        self.n_users = n_users
        self.n_items = n_items
        self.n_cat = n_cat
        self.user_emb_size = user_emb_size
        self.item_emb_size = item_emb_size
        self.batch_size = batch_size
        self.set_hyperparams(hyperparams)

    @property
    def default_hyperparams(self) -> dict:
        """Returns the default hyperparameters needed by model.

        Returns:
            dict: Default hyperparameters
        """
        return {
            "emb_l2_reg_lambda": 1e-5
        }

    def set_hyperparams(self, hyperparams: dict):
        """Sets the hyperparamets as attributes in the model

        Args:
            hyperparams (dict): The user given hyperparams which will override the default_hyperparams
        """
        hyperparams = {**self.default_hyperparams, **hyperparams}
        for hyperparam, value in hyperparams.items():
            setattr(self, hyperparam, value)

    def create_variables(self):
        initializer = tf.initializers.glorot_uniform()
        # Create embeddings
        self.user_embedding = tf.Variable(initializer((self.n_users, self.user_emb_size)), name="user_embedding")
        self.item_embedding = tf.Variable(initializer((self.n_items, self.item_emb_size)), name="item_embedding")

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
