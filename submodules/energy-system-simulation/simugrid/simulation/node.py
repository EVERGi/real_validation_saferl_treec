class Node:
    """
    Node that contains information about all the components for the simulation

    :ivar assets: list of assets in the node
    :type assets: list[Asset]
    :ivar environment: the environment of the node
    :type environment: Environment
    :ivar connected_nodes:
    :type connected_nodes:
    :ivar microgrid: the microgrid
    :type microgrid: Microgrid

    """

    def __init__(self, microgrid):
        self.assets = []
        self.environment = None  # Environment object of the node
        self.connected_nodes = list()
        self.microgrid = microgrid
        microgrid.nodes += [self]

    def set_environment(self, environment):
        self.environment = environment
        if environment not in self.microgrid.environments:
            self.microgrid.environments += [environment]

    def simulate_step(self):
        for asset in self.assets:
            asset.power_reset()
            asset.set_power_limits(self.environment)

    def power_consequences(self, execute=True):
        if execute:
            for asset in self.assets:
                asset.power_consequences()
        else:
            for asset in self.assets:
                new_asset = asset.exec_copy()
                new_asset.power_consequences()

    def get_index(self):
        for i, node in enumerate(self.microgrid.nodes):
            if node == self:
                return i

    @property
    def index(self):
        return self.get_index()

    def get_type_assets(self, asset_type):
        """
        Get all assets of a certain type

        :param asset_type: the type of asset
        :type asset_type: AssetType
        """
        return list(filter(lambda asset: asset.asset_type == asset_type, self.assets))
