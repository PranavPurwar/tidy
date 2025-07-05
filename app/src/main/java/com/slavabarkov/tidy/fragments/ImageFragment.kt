package com.slavabarkov.tidy.fragments

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Intent
import android.database.Cursor
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.core.net.toUri
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.bumptech.glide.Glide
import com.github.chrisbanes.photoview.PhotoView
import com.google.android.material.appbar.MaterialToolbar
import com.slavabarkov.tidy.R
import com.slavabarkov.tidy.utils.ORTAnalyzer
import com.slavabarkov.tidy.viewmodels.ORTImageViewModel
import com.slavabarkov.tidy.viewmodels.SearchViewModel
import java.nio.FloatBuffer
import java.text.DateFormat

class ImageFragment : Fragment() {
    private var imageUri: Uri? = null
    private var imageId: Long? = null
    private val mORTImageViewModel: ORTImageViewModel by activityViewModels()
    private val mSearchViewModel: SearchViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? {
        val view = inflater.inflate(R.layout.fragment_image, container, false)
        val bundle = this.arguments
        bundle?.let {
            imageId = it.getLong("image_id")
            imageUri = it.getString("image_uri")?.toUri()
        }

        //Get image date from image URI
        val cursor: Cursor =
            requireContext().contentResolver.query(imageUri!!, null, null, null, null)!!
        cursor.moveToFirst()
        val idx: Int = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATE_MODIFIED)
        val date: Long = cursor.getLong(idx) * 1000
        cursor.close()

        val toolbar = view.findViewById<MaterialToolbar>(R.id.toolbar)

        toolbar.setTitle(DateFormat.getDateInstance().format(date))

        val singleImageView: PhotoView = view.findViewById(R.id.singeImageView)
        Glide.with(view).load(imageUri).into(singleImageView)

        val buttonImage2Image: Button = view.findViewById(R.id.buttonImage2Image)
        buttonImage2Image.setOnClickListener {
            imageId?.let {
                val imageIndex = mORTImageViewModel.idxList.indexOf(it)
                val imageEmbedding = mORTImageViewModel.embeddingsList[imageIndex]
                mSearchViewModel.sortByCosineDistance(
                    imageEmbedding, mORTImageViewModel.embeddingsList, mORTImageViewModel.idxList
                )
            }
            mSearchViewModel.fromImg2ImgFlag = true
            parentFragmentManager.popBackStack()
        }

        toolbar.setOnMenuItemClickListener {
            when (it.itemId) {
                R.id.action_share -> {
                    val sendIntent: Intent = Intent().apply {
                        action = Intent.ACTION_SEND
                        putExtra(Intent.EXTRA_STREAM, imageUri)
                        type = "image/*"
                    }
                    val shareIntent = Intent.createChooser(sendIntent, null)
                    startActivity(shareIntent)
                    true
                }

                else -> false
            }
        }

        // Load and preprocess the image
        val bitmap = MediaStore.Images.Media.getBitmap(requireContext().contentResolver, imageUri)
        val ortEnvironment = OrtEnvironment.getEnvironment()
        val modelBytes = requireContext().assets.open("mobilenet_v3_float.onnx").readBytes()
        val ortSession = ortEnvironment.createSession(modelBytes)
        val labels = requireContext().assets.open("classes.txt").bufferedReader().readLines()
        val labelScoresTextView = view.findViewById<android.widget.TextView>(R.id.labelScoresTextView)
        val analyzer = ORTAnalyzer(ortSession) { result ->
            val sb = StringBuilder()
            result.detectedIndices.forEachIndexed { index, i ->
                if (i < labels.size) {
                    sb.append("Label: ")
                    sb.append(labels[i])
                    sb.append(", Score: ")
                    sb.append(String.format(java.util.Locale.US, "%.2f", result.detectedScore[index]))
                    sb.append("\n")
                }
            }
            if (sb.isEmpty()) {
                sb.append("No labels detected.")
            }
            labelScoresTextView.post {
                labelScoresTextView.text = sb.toString()
            }
        }

        analyzer.analyze(bitmap)

        /*
                val inputImage = InputImage.fromFilePath(requireContext(), imageUri!!)

                val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
                labeler.process(inputImage)
                    .addOnSuccessListener { labels ->
                        for (label in labels) {
                            val text = label.text
                            val confidence = label.confidence
                            val index = label.index

                            Log.d("ImageFragment", "Label: $text, Confidence: $confidence, Index: $index")
                        }
                    }
                    .addOnFailureListener { e ->
                        e.printStackTrace()
                    }

                re*/
        return view
    }

    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        val floatBuffer = FloatBuffer.allocate(1 * 3 * 640 * 640)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        for (y in 0 until 640) {
            for (x in 0 until 640) {
                val pixel = bitmap.getPixel(x, y)
                val r = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
                val g = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
                val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
                floatBuffer.put(r)
                floatBuffer.put(g)
                floatBuffer.put(b)
            }
        }
        floatBuffer.rewind()
        return OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            floatBuffer,
            longArrayOf(1, 3, 640, 640)
        )
    }

    private fun getTopResults(outputArray: FloatArray, topK: Int): List<Pair<String, Float>> {
        val labels = requireContext().assets.open("labels.txt").bufferedReader().readLines()
        val results = mutableListOf<Pair<String, Float>>()

        for (i in outputArray.indices step 6) {
            val score = outputArray[i + 4]
            Log.d("getTopResults", "Score: $score, Index: $i")
            if (score > 0.5) {
                val id = outputArray[i + 5].toInt()
                val label = labels.getOrNull(id) ?: "Unknown"
                results.add(label to score)
                Log.d("getTopResults", "Label: $label, Score: $score")
            }
        }

        return results.sortedByDescending { it.second }.take(topK)
    }
}
