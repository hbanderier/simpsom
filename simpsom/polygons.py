    @staticmethod
    def distance_pbc(
        node_a: np.ndarray,
        node_b: np.ndarray,
        net_shape: Tuple[float],
        distance_func: Callable,
        axis: Union[int, None] = None,
        xp: ModuleType = np,
    ) -> float:
        """Manage distances with PBC based on the tiling.

        Args:
            node_a (np.ndarray): the first node
                from which the distance will be calculated.
            node_b (np.ndarray): the second node
                from which the distance will be calculated.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.
            axis (int): axis along which the minimum
                distance across PBC will be calculated.
            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (float): the distance adjusted by PBC.
        """

        net_shape = xp.array((net_shape[0], net_shape[1]))

        return xp.min(
            xp.array(
                (
                    distance_func(node_a, node_b),
                    distance_func(node_a, node_b + net_shape * xp.array((1, 0))),
                    distance_func(node_a, node_b - net_shape * xp.array((1, 0))),
                    distance_func(node_a, node_b + net_shape * xp.array((0, 1))),
                    distance_func(node_a, node_b - net_shape * xp.array((0, 1))),
                    distance_func(node_a, node_b + net_shape),
                    distance_func(node_a, node_b - net_shape),
                    distance_func(node_a, node_b + net_shape * xp.array((-1, 1))),
                    distance_func(node_a, node_b - net_shape * xp.array((-1, 1))),
                )
            ),
            axis=0,
        )

    @staticmethod
    def neighborhood_pbc(
        center_node: Tuple[np.ndarray],
        nodes: Tuple[np.ndarray],
        net_shape: Tuple[float],
        distance_func: Callable,
        xp: ModuleType = np,
    ) -> np.ndarray:
        """Manage neighborhood with PBC based on the tiling, adapted for
        batch training neighborhood functions. Works along a single
        provided axis and calculates the distance of a single node (center_node) from
        all other nodes in the network (nodes)

        Args:
            center_node (Tuple[np.ndarray]): position (index) of the first node along
                the provided axis. Shaped as (net_shape[1], 1, 1), for each axis.
            nodes (Tuple[np.ndarray]): the position of all nodes
                long a given axis as a matrix.
                Shaped as (1, net_shape[1], net_shape[0]), for each axis.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.

            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (np.ndarray): the distance from all nodes adjusted by PBC.
        """

        net_shape = (
            xp.full(nodes[0].shape, fill_value=net_shape[0]),
            xp.full(nodes[1].shape, fill_value=net_shape[1]),
        )

        return xp.max(
            xp.array(
                (
                    distance_func(center_node[0], nodes[0]),
                    distance_func(center_node[0], nodes[0] + net_shape[0]),
                    distance_func(center_node[0], nodes[0] - net_shape[0]),
                )
            ),
            axis=0,
        ), xp.max(
            xp.array(
                (
                    distance_func(center_node[1], nodes[1]),
                    distance_func(center_node[1], nodes[1] + net_shape[1]),
                    distance_func(center_node[1], nodes[1] - net_shape[1]),
                )
            ),
            axis=0,
        )

    @staticmethod
    def distance_pbc(
        node_a: np.ndarray,
        node_b: np.ndarray,
        net_shape: Tuple[float],
        distance_func: Callable,
        axis: Union[int, None] = None,
        xp: ModuleType = np,
    ) -> float:
        """Manage distances with PBC based on the tiling.

        Args:
            node_a (np.ndarray): the first node
                from which the distance will be calculated.
            node_b (np.ndarray): the second node
                from which the distance will be calculated.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.
            axis (int): axis along which the minimum
                distance across PBC will be calculated.
            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (float): the distance adjusted by PBC.
        """

        offset = 0 if net_shape[1] % 2 == 0 else 0.5
        offset = xp.array((offset, 0))
        net_shape = xp.array((net_shape[0], net_shape[1] * 2 / np.sqrt(3) * 3 / 4))

        return xp.min(
            xp.array(
                (
                    distance_func(node_a, node_b),
                    distance_func(node_a, node_b + net_shape * xp.array((1, 0))),
                    distance_func(node_a, node_b - net_shape * xp.array((1, 0))),
                    distance_func(
                        node_a, node_b + net_shape * xp.array((0, 1)) + offset
                    ),
                    distance_func(
                        node_a, node_b - net_shape * xp.array((0, 1)) - offset
                    ),
                    distance_func(node_a, node_b + net_shape + offset),
                    distance_func(node_a, node_b - net_shape - offset),
                    distance_func(
                        node_a, node_b + net_shape * xp.array((-1, 1)) + offset
                    ),
                    distance_func(
                        node_a, node_b - net_shape * xp.array((-1, 1)) - offset
                    ),
                )
            ),
            axis=axis,
        )

    @staticmethod
    def neighborhood_pbc(
        center_node: Tuple[np.ndarray],
        nodes: Tuple[np.ndarray],
        net_shape: Tuple[float],
        distance_func: Callable,
        xp: ModuleType = np,
    ) -> np.ndarray:
        """Manage neighborhood with PBC based on the tiling, adapted for
        batch training neighborhood functions. Works along a single
        provided axis and calculates the distance of a single node (center_node) from
        all other nodes in the network (nodes)

        Args:
            center_node (Tuple[np.ndarray]): position (index) of the first node along
                the provided axis. Shaped as (net_shape[1], 1, 1), for each axis.
            nodes (Tuple[np.ndarray]): the position of all nodes
                long a given axis as a matrix.
                Shaped as (1, net_shape[1], net_shape[0]), for each axis.
            net_shape (tuple[float, float]): the sizes of
                the network.
            distance_func (function): the function
                to calculate distance between nodes.
            xp (numpy or cupy): the numeric library
                to handle arrays.

        Returns:
            (np.ndarray): the distance from all nodes adjusted by PBC.
        """

        offset = xp.full(nodes[0].shape, fill_value=0)
        if net_shape[1] % 2 != 0:
            offset[:] = 0.5

        net_shape = (
            xp.full(nodes[0].shape, fill_value=net_shape[0]),
            xp.full(nodes[1].shape, fill_value=net_shape[1] * 2 / xp.sqrt(3) * 3 / 4),
        )

        return xp.max(
            xp.array(
                (
                    distance_func(center_node[0], nodes[0]),
                    distance_func(center_node[0], nodes[0] + net_shape[0] + offset),
                    distance_func(center_node[0], nodes[0] - net_shape[0] - offset),
                )
            ),
            axis=0,
        ), xp.max(
            xp.array(
                (
                    distance_func(center_node[1], nodes[1]),
                    distance_func(center_node[1], nodes[1] + net_shape[1]),
                    distance_func(center_node[1], nodes[1] - net_shape[1]),
                )
            ),
            axis=0,
        )
