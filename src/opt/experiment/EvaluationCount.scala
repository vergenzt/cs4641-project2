package opt.experiment

import opt.OptimizationProblem
import shared.Instance

trait EvaluationCount extends OptimizationProblem {
  this: OptimizationProblem =>

  private var _count = 0

  def evaluationCount: Int = _count

  abstract override def value(x: Instance): Double = {
    _count += 1
    super.value(x)
  }

}
