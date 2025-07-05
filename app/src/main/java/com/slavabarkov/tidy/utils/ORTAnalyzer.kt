package com.slavabarkov.tidy.utils

import com.slavabarkov.tidy.preProcess

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import java.util.*
import kotlin.math.exp
import androidx.core.graphics.scale


internal data class Result(
    var detectedIndices: List<Int> = emptyList(),
    var detectedScore: MutableList<Float> = mutableListOf(),
    var processTimeMs: Long = 0
)

internal class ORTAnalyzer(
    private val ortSession: OrtSession?,
    private val callBack: (Result) -> Unit
) {

    // Get index of top 3 values
    // This is for demo purpose only, there are more efficient algorithms for topK problems
    private fun getTop3(labelVals: FloatArray): List<Int> {
        val indices = mutableListOf<Int>()
        for (k in 0..2) {
            var max = 0.0f
            var idx = 0
            for (i in 0..labelVals.size - 1) {
                val label_val = labelVals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    // Calculate the SoftMax for the input array
    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    fun analyze(imgBitmap: Bitmap) {
        val rawBitmap = imgBitmap.scale(224, 224, false)
        val bitmap = rawBitmap.rotate(0f)

        var result = Result()
        val imgData = preProcess(bitmap)
        val inputName = ortSession?.inputNames?.iterator()?.next()
        val shape = longArrayOf(1, 3, 224, 224)
        val env = OrtEnvironment.getEnvironment()
        env.use {
            val tensor = OnnxTensor.createTensor(env, imgData, shape)
            val startTime = SystemClock.uptimeMillis()
            var abandoned = false
            tensor.use {
                val output = try {
                    ortSession?.run(Collections.singletonMap(inputName, tensor))
                } catch (e: Exception) {
                    null
                }
                val elapsed = SystemClock.uptimeMillis() - startTime
                if (elapsed > 5000) {
                    abandoned = true
                }
                output?.use {
                    if (!abandoned) {
                        result.processTimeMs = elapsed
                        @Suppress("UNCHECKED_CAST")
                        val rawOutput = ((output.get(0)?.value) as Array<FloatArray>)[0]
                        val probabilities = softMax(rawOutput)
                        result.detectedIndices = getTop3(probabilities)
                        for (idx in result.detectedIndices) {
                            result.detectedScore.add(probabilities[idx])
                        }
                    }
                }
            }
            if (abandoned) {
                result = Result(
                    detectedIndices = emptyList(),
                    detectedScore = mutableListOf(),
                    processTimeMs = 0
                )
                // You can add a message field to Result if needed, or handle in callback
            }
        }
        callBack(result)
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}
