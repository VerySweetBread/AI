import kotlin.math.pow
import kotlin.properties.Delegates

class Perceptron (private val layers: Array<Layer>, private val k: Double = 0.5) {
    val input_layer: Layer
    val output_layer: Layer

    init {
        input_layer = layers.first()
        output_layer = layers.last()

        layers.mapIndexed { index, layer -> layer.position = index }
        for (i in 0..layers.size-2) {
            println(i)
            Weight(layers[i], layers[i+1])
        }
        this.count()
    }

    private fun count() { layers.forEach { it.count() }}

    fun input(array: Array<Double>) {
        input_layer.nodes.mapIndexed { index, node ->
            if (node !is Bias) node.valuE = array[index]
        }
        this.count()
    }

    private fun backPropagation(input: Array<Double>) {
        output_layer.nodes.mapIndexed { index, node -> node.error = input[index] - node.valuE }
        for (layer in layers.drop(1).dropLast(1).reversed()) {
            for (node in layer.nodes) {
                node.error = 0.toDouble()
                for (n_node in node.nextNodes) {
                    node.error += n_node.error * layer.nextWeight!!.weight[listOf(node, n_node)]!!
                }
            }
        }
        for (layer in layers.dropLast(1)) {
            for (node in layer.nodes) {
                for (n_node in node.nextNodes) {
                    node.parent.nextWeight!!.weight[listOf(node, n_node)] =
                        node.parent.nextWeight!!.weight[listOf(node, n_node)]!! +
                                k * n_node.error * ActivationFun.logisticDerivative(n_node.valuE)*node.valuE
                }
            }
        }
    }

    fun teach(sets: Array<Array<Array<Double>>>, epochs: Int = 100000, silent: Boolean = true) {
        for (epoch in 1..epochs) {
            if (!silent) println("epoch #$epoch")
            for (set in sets) {
                this.input(set[0])
                this.backPropagation(set[1])
            }
        }
    }

    fun output(): List<Double> {
        var output = listOf<Double>()
        for (node in layers.last().nodes) { output += node.valuE }
        return output
    }

    fun output_int(): List<Int> {
        var output = listOf<Int>()
        for (node in layers.last().nodes) { output += if (node.valuE > 0.5) 1 else 0 }
        return output
    }
}

open class Node(val parent: Layer, private val position: Int) {
    open var valuE = 0.toDouble()
    var error = 0.toDouble()
    var prevNodes = listOf<Node>()
    var nextNodes = listOf<Node>()

    open fun getValue(): Double {
        return if (prevNodes.isEmpty()) valuE
        else {
            valuE = 0.toDouble()
            for (node in prevNodes) {
                valuE += node.valuE * parent.prevWeight!!.weight[listOf(node, this)]!!
            }
//            tmp = value
            valuE = ActivationFun.logistic(valuE)
            valuE
        }
    }

    override fun toString(): String {
        return "[${parent.position}:${position} | $valuE ]"
    }
}

class Bias(parent: Layer, position: Int) : Node(parent, position) {
    override var valuE = 1.toDouble()
    override fun getValue() = 1.toDouble()
}

open class Layer(amount: Int, bias: Boolean = true) {
    var position by Delegates.notNull<Int>()
    var nodes = listOf<Node>()
    var nextWeight: Weight? = null
    var prevWeight: Weight? = null

    init {
        for (i in 0 until amount) {
            nodes += Node(this, i)
        }
         if (bias) nodes += Bias(this, nodes.size)
    }

    fun count() {
        for (node in nodes) { node.getValue() }
    }
}

class Weight(prev_: Layer, next_: Layer) {
    var weight = HashMap<List<Node>, Double>()

    init {
        prev_.nextWeight = this
        next_.prevWeight = this

        for (input_node in prev_.nodes) {
            for (output_node in next_.nodes) {
                if (output_node is Bias) continue
                weight[listOf(input_node, output_node)] = Math.random()
                input_node.nextNodes += output_node
                output_node.prevNodes += input_node
            }
        }

        println(weight)
    }
}

class ActivationFun {
    companion object{
        fun logistic(x: Double) = 1/(1 + Math.E.pow(-x))
        fun logisticDerivative(x: Double) = x * (1 - x)
    }
}
