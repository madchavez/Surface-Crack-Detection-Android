package com.example.thesisapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.thesisapp.ml.ConvertedModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp

class MainActivity : ComponentActivity() {

    lateinit var imageView: ImageView
    lateinit var btn: Button
    lateinit var camBtn: Button
    lateinit var predictBtn: Button
    lateinit var bitmap: Bitmap
    lateinit var textView: TextView
    private lateinit var result: TextView
    lateinit var labels: List<String>
    lateinit var model: ConvertedModel
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(120,120, ResizeOp.ResizeMethod.BILINEAR)).add(TransformToGrayscaleOp()).build()


    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val requestPermission =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
                // do something
            }

        btn = findViewById(R.id.selectBtn)
        camBtn = findViewById(R.id.cptBtn)
        predictBtn = findViewById(R.id.predict)
        imageView = findViewById(R.id.imageView)
        textView = findViewById(R.id.result)

        val model = ConvertedModel.newInstance(this)

        btn.setOnClickListener {
            val intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent, 101)

        }
        camBtn.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                // Pass any permission you want while launching
                requestPermission.launch(Manifest.permission.CAMERA)
            }
            val camIntent = Intent()
            camIntent.setAction(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(camIntent, 102)
        }

        predictBtn.setOnClickListener {
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 120, 120, 1), DataType.FLOAT32)
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            val byteBuffer = tensorImage.buffer
            inputFeature0.loadBuffer(byteBuffer)
// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val classed = outputFeature0.floatArray[0].toString()
            if(classed == "0.0"){
                textView.setText("Structure is Cracked.")
            }
            else{
                textView.setText("Structure is Not Cracked.")
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101) {
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        }
        else if (requestCode == 102){
            if (data != null) {
                bitmap = data.getExtras()!!.get("data") as Bitmap
                imageView.setImageBitmap(bitmap)
            }
        }
    }
}

