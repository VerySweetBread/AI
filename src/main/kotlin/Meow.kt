fun main() {
    val model = Perceptron(3, arrayOf(), 2)
    model.teach(
        arrayOf(
            arrayOf(arrayOf(0.0, 0.0, 1.0), arrayOf(0.0, 0.0)),
            arrayOf(arrayOf(0.0, 1.0, 0.0), arrayOf(1.0, 0.0)),
            arrayOf(arrayOf(1.0, 0.0, 1.0), arrayOf(0.0, 1.0)),
            arrayOf(arrayOf(1.0, 1.0, 1.0), arrayOf(1.0, 1.0))
        ), 10000
    )
    model.input(arrayOf(0.0, 0.0, 1.0)); println(model.output())
    model.input(arrayOf(0.0, 1.0, 1.0)); println(model.output())
    model.input(arrayOf(1.0, 0.0, 0.0)); println(model.output())
    model.input(arrayOf(1.0, 1.0, 0.0)); println(model.output())
}