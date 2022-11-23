import kotlin.math.pow

class Perceptron (val input_n: Int, val hidden_n: Array<Int> = arrayOf(), val output_n: Int, val k: Double = 0.5) {
    val layers: List<Layer>

    init {
        var tmp_layers = listOf<Layer>()

        tmp_layers += Layer(input_n, 0)

        for (layer_n in hidden_n) {
            for (layer in 0 until layer_n) {
                tmp_layers += Layer(layer, layer+1)
            }
        }

        tmp_layers += Layer(output_n, tmp_layers.size)
        tmp_layers.last().nodes = tmp_layers.last().nodes.dropLast(1)

        layers = tmp_layers

        for (i in layers.indices-1) {
            println(i)
            Weight(layers[i], layers[i+1])
        }
        this.count()
    }

    fun count() {
        for (layer in layers) { layer.count() }
    }

    fun input(array: Array<Double>) {
        for (index in 0 until layers[0].nodes.dropLast(1).size) {
            layers[0].nodes[index].value = array[index]
        }
        this.count()
    }

    fun back_propo(input: Array<Double>) {
        for (i in input.indices) { layers.last().nodes[i].error = input[i] - layers.last().nodes[i].value }
        for (layer in layers.dropLast(1).reversed()) {
            for (node in layer.nodes) {
                node.error = 0.toDouble()
                for (n_node in node.next_nodes) {
                    node.error += n_node.error * layer.next_weight!!.weight[listOf(node, n_node)]!!
                }
            }
        }
        for (layer in layers.dropLast(1)) {
            for (node in layer.nodes) {
                for (n_node in node.next_nodes) {
                    node.parent.next_weight!!.weight[listOf(node, n_node)] =
                        node.parent.next_weight!!.weight[listOf(node, n_node)]!! +
                                k * n_node.error * activation_fun.logistic_(n_node.value)*node.value
                }
            }
        }
    }

    fun teach(sets: Array<Array<Array<Double>>>, epochs: Int = 100000) {
        for (epoch in 1..epochs) {
            println("epoch #$epoch")
            for (set in sets) {
                this.input(set[0])
                this.back_propo(set[1])
            }
        }
    }

    fun output(): List<Double> {
        var output = listOf<Double>()
        for (node in layers.last().nodes) { output += node.value }
        return output
    }
}

open class Node(val parent: Layer, val position: Int) {
    open var value = 0.toDouble()
    var error = 0.toDouble()
    //    var tmp = 0.toDouble()
    var prev_nodes = listOf<Node>()
    var next_nodes = listOf<Node>()

    open fun get_value(): Double {
        return if (prev_nodes.isEmpty()) value
        else {
            value = 0.toDouble()
            for (node in prev_nodes) {
                value += node.value * parent.prev_weight!!.weight[listOf(node, this)]!!
            }
//            tmp = value
            value = activation_fun.logistic(value)
            value
        }
    }

    override fun toString(): String {
        return "[${parent.position}:${position} | $value ]"
    }
}

class Bias(parent: Layer, position: Int) : Node(parent, position) {
    override var value = 1.toDouble()
    override fun get_value() = 1.toDouble()
}

class Layer(amount: Int, val position: Int) {
    var nodes = listOf<Node>()
    var next_weight: Weight? = null
    var prev_weight: Weight? = null

    init {
        for (i in 0 until amount) {
            nodes += Node(this, i)
        }
        nodes += Bias(this, nodes.size)
    }

    fun count() {
        for (node in nodes) { node.get_value() }
    }
}

class Weight(prev_: Layer, next_: Layer) {
    var weight = HashMap<List<Node>, Double>()

    init {
        prev_.next_weight = this
        next_.prev_weight = this

        for (input_node in prev_.nodes) {
            for (output_node in next_.nodes) {
                if (output_node is Bias) continue
                weight[listOf(input_node, output_node)] = Math.random()
                input_node.next_nodes += output_node
                output_node.prev_nodes += input_node
            }
        }

        println(weight)
    }
}

class activation_fun {
    companion object{
        fun logistic(x: Double) = 1/(1 + Math.E.pow(-x))
        fun logistic_(x: Double) = x * (1 - x)
    }
}

enum class Functions {
    Logistic
}