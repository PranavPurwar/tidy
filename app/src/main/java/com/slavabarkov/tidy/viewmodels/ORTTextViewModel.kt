/**
 * Copyright 2023 Viacheslav Barkov
 */

package com.slavabarkov.tidy.viewmodels

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.util.JsonReader
import androidx.lifecycle.AndroidViewModel
import com.slavabarkov.tidy.R
import com.slavabarkov.tidy.normalizeL2
import com.slavabarkov.tidy.tokenizer.ClipTokenizer
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.HashMap

class ORTTextViewModel(application: Application) : AndroidViewModel(application) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val assets = getApplication<Application>().assets
    // OpenCLIP model
    private val model = assets.open("onnx32_textual.onnx").readBytes()
    private val session = ortEnv.createSession(model)

    private val tokenizerVocab: Map<String, Int> = getVocab()
    private val tokenizerMerges: HashMap<Pair<String, String>, Int> = getMerges()
    private val tokenBOS: Int = 49406
    private val tokenEOS: Int = 49407
    private val tokenizer = ClipTokenizer(tokenizerVocab, tokenizerMerges)

    private val queryFilter = Regex("[^A-Za-z0-9 ]")

    fun getTextEmbedding(text: String): FloatArray {
        // Tokenize
        val textClean = queryFilter.replace(text, "").lowercase()
        var tokens: MutableList<Long> = ArrayList()
        tokens.add(tokenBOS.toLong())
        tokens.addAll(tokenizer.encode(textClean).map { it.toLong() })
        tokens.add(tokenEOS.toLong())

        var mask: MutableList<Long> = ArrayList()
        for (i in 0 until tokens.size) {
            mask.add(1L)
        }
        while (tokens.size < 77) {
            tokens.add(0L)
            mask.add(0L)
        }
        tokens = tokens.subList(0, 77)
        mask = mask.subList(0, 77)

        // Convert to tensor
        val inputShape = longArrayOf(1, 77)
        val inputIds = LongBuffer.allocate(1 * 77)
        inputIds.rewind()
        for (i in 0 until 77) {
            inputIds.put(tokens[i])
        }
        inputIds.rewind()
        val inputIdsTensor = OnnxTensor.createTensor(ortEnv, inputIds, inputShape)

        val attentionMask = LongBuffer.allocate(1 * 77)
        attentionMask.rewind()
        for (i in 0 until 77) {
            attentionMask.put(mask[i])
        }
        attentionMask.rewind()
        val attentionMaskTensor = OnnxTensor.createTensor(ortEnv, attentionMask, inputShape)

        val inputMap: MutableMap<String, OnnxTensor> = HashMap()
        inputMap["input"] = inputIdsTensor

        val output = session?.run(inputMap)
        output.use {
            @Suppress("UNCHECKED_CAST") var rawOutput =
                ((output?.get(0)?.value) as Array<FloatArray>)[0]
            rawOutput = normalizeL2(rawOutput)
            return rawOutput
        }
    }

    fun getVocab(): Map<String, Int> {
        val vocab = hashMapOf<String, Int>().apply {
            assets.open("vocab.json").use {
                val vocabReader = JsonReader(InputStreamReader(it, "UTF-8"))
                vocabReader.beginObject()
                while (vocabReader.hasNext()) {
                    val key = vocabReader.nextName().replace("</w>", " ")
                    val value = vocabReader.nextInt()
                    put(key, value)
                }
                vocabReader.close()
            }
        }
        return vocab
    }

    fun getMerges(): HashMap<Pair<String, String>, Int> {
        val merges = hashMapOf<Pair<String, String>, Int>().apply {
            assets.open("merges.txt").use {
                val mergesReader = BufferedInputStream(it).bufferedReader()
                mergesReader.useLines { seq ->
                    seq.drop(1).forEachIndexed { i, s ->
                        val list = s.split(" ")
                        val keyTuple = list[0] to list[1].replace("</w>", " ")
                        put(keyTuple, i)
                    }
                }
            }
        }
        return merges
    }
}
