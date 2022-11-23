import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import java.io.File

var sepal_length_max = 0.0
var sepal_width_max = 0.0
var petal_length_max = 0.0
var petal_width_max = 0.0

fun main() {
    val file = File("/run/media/sweetbread/50AF29954CE66E9F/Coding/Kotlin/AI/src/main/resources/IRIS.csv")
    val data: List<Iris> = csvReader().readAll(file).drop(1).map { Iris(it[0].toDouble(), it[1].toDouble(), it[2].toDouble(), it[3].toDouble(), it[4]) } .shuffled()
    data.forEach {
        if (it.sepal_length > sepal_length_max) sepal_length_max = it.sepal_length
        if (it.sepal_width  > sepal_width_max)  sepal_width_max  = it.sepal_width
        if (it.petal_length > petal_length_max) petal_length_max = it.petal_length
        if (it.petal_width  > petal_width_max)  petal_width_max  = it.petal_width
    }

    val model = Perceptron(arrayOf(
        Layer(4),
        Layer(4),
        Layer(3, false)
    ))
    println(model.output())

    model.teach(data.subList(0, 50).map { it.data_field() }.toTypedArray(), 100, false)
    data[100].data_field()[1].forEach { print(it.toInt()); print(" ") }
    println()
    model.input(data[100].data_field()[0]); println(model.output())
}

class Iris (
    val sepal_length: Double,
    val sepal_width: Double,
    val petal_length: Double,
    val petal_width: Double,
    val species: String
    ) {
    fun data_field(): Array<Array<Double>> {
        return arrayOf(
            arrayOf(sepal_length/sepal_length_max, sepal_width/sepal_width_max, petal_length/petal_length_max, petal_width/petal_width_max),
            arrayOf(
                if (species == "Iris-setosa") 1.0 else 0.0,
                if (species == "Iris-versicolor") 1.0 else 0.0,
                if (species == "Iris-virginica") 1.0 else 0.0
            )
        )
    }
}