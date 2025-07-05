package com.slavabarkov.tidy.viewmodels

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context.NOTIFICATION_SERVICE
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.core.app.NotificationCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.slavabarkov.tidy.MainActivity
import com.slavabarkov.tidy.R
import com.slavabarkov.tidy.centerCrop
import com.slavabarkov.tidy.data.ImageEmbedding
import com.slavabarkov.tidy.data.ImageEmbeddingDatabase
import com.slavabarkov.tidy.data.ImageEmbeddingRepository
import com.slavabarkov.tidy.normalizeL2
import com.slavabarkov.tidy.preProcess
import kotlinx.coroutines.launch
import java.util.Collections

class ORTImageViewModel(application: Application) : AndroidViewModel(application) {
    private var repository: ImageEmbeddingRepository
    var idxList: ArrayList<Long> = arrayListOf()
    var embeddingsList: ArrayList<FloatArray> = arrayListOf()
    var progress: MutableLiveData<Double> = MutableLiveData(0.0)
    private var notificationManager = application.getSystemService(NOTIFICATION_SERVICE) as NotificationManager
    private var builder: NotificationCompat.Builder

    init {
        createNotificationChannel()

        val pendingIntent = Intent(application, MainActivity::class.java).let { notificationIntent ->
            PendingIntent.getActivity(application, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE)
        }
        builder = NotificationCompat.Builder(application, "progress_channel")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Image Processing")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setStyle(NotificationCompat.BigTextStyle())
            .setContentIntent(pendingIntent)
            .setAutoCancel(false)
        builder.setProgress(100, 0, false)
        notificationManager.notify(1, builder.build())

        val imageEmbeddingDao = ImageEmbeddingDatabase.getDatabase(application).imageEmbeddingDao()
        repository = ImageEmbeddingRepository(imageEmbeddingDao)
    }


    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            "progress_channel",
            "Progress Channel",
            NotificationManager.IMPORTANCE_LOW
        )
        notificationManager.createNotificationChannel(channel)
    }

    fun generateIndex() {
        val assets = getApplication<Application>().assets
        val model = assets.open("onnx32_visual.onnx").readBytes()
        val ortEnv = OrtEnvironment.getEnvironment()
        val session = ortEnv.createSession(model)
        viewModelScope.launch {

            val uri: Uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            val projection = arrayOf(
                MediaStore.Images.Media._ID,
                MediaStore.Images.Media.DATE_MODIFIED,
                MediaStore.Images.Media.BUCKET_DISPLAY_NAME
            )
            val sortOrder = "${MediaStore.Images.Media._ID} ASC"
            val contentResolver = getApplication<Application>().contentResolver
            val cursor = contentResolver.query(uri, projection, null, null, sortOrder)
            val totalImages = cursor?.count ?: 0
            cursor?.use {
                val idColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val dateColumn: Int =
                    it.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
                val bucketColumn: Int =
                    it.getColumnIndex(MediaStore.Images.Media.BUCKET_DISPLAY_NAME)
                while (it.moveToNext()) {
                    val position = it.position
                    try {
                        val id: Long = it.getLong(idColumn)
                        val date: Long = it.getLong(dateColumn)
                        val bucket = it.getString(bucketColumn)
                        if (bucket == "Screenshots") {
                            progress.postValue((position + 1).toDouble() / totalImages)
                            updateNotification(position + 1, totalImages)
                            continue
                        }
                        val record = repository.getRecord(id) as ImageEmbedding?
                        if (record != null) {
                            idxList.add(id)
                            embeddingsList.add(record.embedding)
                        } else {
                            val imageUri: Uri = Uri.withAppendedPath(uri, id.toString())
                            if (imageUri == Uri.EMPTY) {
                                progress.postValue((position + 1).toDouble() / totalImages)
                                updateNotification(position + 1, totalImages)
                                continue
                            }
                            val inputStream = try {
                                contentResolver.openInputStream(imageUri)
                            } catch (e: Exception) {
                                null
                            }
                            if (inputStream == null) {
                                progress.postValue((position + 1).toDouble() / totalImages)
                                updateNotification(position + 1, totalImages)
                                continue
                            }
                            val bytes = inputStream.readBytes()
                            inputStream.close()
                            val bitmap: Bitmap? = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                            if (bitmap == null) {
                                progress.postValue((position + 1).toDouble() / totalImages)
                                updateNotification(position + 1, totalImages)
                                continue
                            }
                            val rawBitmap = centerCrop(bitmap, 224)
                            val inputShape = longArrayOf(1, 3, 224, 224)
                            val inputName = "input"
                            val imgData = preProcess(rawBitmap)
                            val inputTensor = OnnxTensor.createTensor(ortEnv, imgData, inputShape)
                            inputTensor.use {
                                val output = session?.run(Collections.singletonMap(inputName, inputTensor))
                                output.use {
                                    @Suppress("UNCHECKED_CAST")
                                    var rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                                    rawOutput = normalizeL2(rawOutput)
                                    repository.addImageEmbedding(ImageEmbedding(id, date, rawOutput))
                                    idxList.add(id)
                                    embeddingsList.add(rawOutput)
                                }
                            }
                        }
                    } catch (e: Exception) {
                        // Optionally log the error for debugging
                    } finally {
                        progress.postValue((position + 1).toDouble() / totalImages)
                        updateNotification(position + 1, totalImages)
                    }
                }
            }
            cursor?.close()
            session.close()
        }
    }

    private fun updateNotification(progress: Int, total: Int) {
        builder.setContentText("Processed $progress/$total images")
        builder.setProgress(total, progress, false)

        if (progress == total) {
            builder.setContentText("Image Processing complete")
                .setProgress(0, 0, false)
        }
        notificationManager.notify(1, builder.build())
    }

}
