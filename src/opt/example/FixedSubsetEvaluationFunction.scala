package opt.example

import opt.EvaluationFunction
import shared.Instance
import scala.util.Random

class FixedSubsetEvaluationFunction(val n: Int, val percentFixed: Double) extends EvaluationFunction {

  val fixedBits: Seq[Option[Int]] = Seq.fill(n) {
    if (Random.nextDouble < percentFixed)
      Some(Random.nextInt(2))
    else
      None
  }

  def value(d: Instance): Double = {
    (0 until n).map({ i =>
      val actual = d.getDiscrete(i)
      val needed = fixedBits(i)
      needed match {
        // add 1 if the actual bit == the fixed bit value
        case Some(`actual`) => 1
        case _ => 0
      }
    }).sum
  }
}
