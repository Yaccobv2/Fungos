package com.example.fungosapp

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    val CAMERA_PERMISSION_REQUEST_CODE = 1000
    val CAMERA_REQUEST_CODE = 10001
     val BATCH_SIZE = 1
    lateinit var bitmap :Bitmap
    lateinit var imgview : ImageView
    lateinit var interpreter: Interpreter
    lateinit var classify: Button
    lateinit var select: Button
    lateinit var takephoto: Button
    lateinit var listview: ListView
    lateinit var arrayadapter: ArrayAdapter<String>
     var shroomname : String = "template"
    lateinit var txtview: TextView
     var isImageLoaded : Boolean = false

    var output = Array(1) {
        FloatArray(
            5
        )}
    var outputConverted = Array( 5){"uwu"}

    fun round(float: Float): Float {
        var multiplier = 100000;
        var temp = 0;
        temp = (float*multiplier).toInt()
        var temp2 = temp.toFloat()/multiplier
        return temp2
    }

    private fun convertOutputToString():Array<String>
    {
        var outputConv = Array( 5){"uwu"}

        for(i in 0..4)
        {
            var temp = round(output[0][i])
          outputConv[i] = temp.toString()
            if(i == 0)
            {
                outputConv[i] = "Muchomor sromotnikowy: " + outputConv[i]
            }
            if(i == 1)
            {
                outputConv[i] = "Muchomor czerwony: " + outputConv[i]
            }
            if(i == 2)
            {
                outputConv[i] = "Borowik szlachetny: " + outputConv[i]
            }
            if(i == 3)
            {
                outputConv[i] = "Pieprznik: " + outputConv[i]
            }
            if(i == 4)
            {
                outputConv[i] = "Gołąbek fiołkowonogi: " + outputConv[i]
            }

      }

        return outputConv
    }

    private fun chooseMax(){

        var maxIndex : Int = 0
        var maxValue : Float = 0.0F
        for(i in 0..4)
        {
            var tag = "Class[" + i + "]"
            Log.d(tag, output[0][i].toString())
            if(output[0][i]>maxValue)
            {
                maxValue = output[0][i]
                maxIndex = i
            }

        }
        if(maxIndex == 0)
        {
            shroomname = "Muchomor sromotnikowy - NIEJADALNE"
        }

        if(maxIndex == 1)
        {
            shroomname = "Muchomor czerwony - NIEJADALNE"
        }
        if(maxIndex == 2)
        {
            shroomname = "Borowik szlachetny - JADALNE"
        }
        if(maxIndex == 3)
        {
            shroomname = "Pieprznik - JADALNE"
        }
        if(maxIndex == 4)
        {
            shroomname = "Gołąbek fiołkowonogi - NIEJADALNE"
        }

    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        setContentView(R.layout.activity_main)
        select = findViewById(R.id.choose_picture)
        classify = findViewById(R.id.bt_classify)
        imgview = findViewById(R.id.iv_capture)
        listview = findViewById(R.id.classes)
        txtview = findViewById(R.id.textView)
        takephoto = findViewById(R.id.bt_take_picture)


        select.setOnClickListener(View.OnClickListener {
            Log.d("mssg", "button pressed")
            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 100)
        })

        takephoto.setOnClickListener(View.OnClickListener {
           val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)

            if(takePictureIntent.resolveActivity(this.packageManager) != null) {
                startActivityForResult(takePictureIntent, 101)
            }
            else
            {
                Toast.makeText(this, "unable to open camera",Toast.LENGTH_SHORT).show()
            }
        })


        classify.setOnClickListener(View.OnClickListener {
            if(isImageLoaded == false)
            {Log.d("Error", "No image attached")}
            else{
            var resized = Bitmap.createScaledBitmap(bitmap, 50, 50, true)
            interpreter = Interpreter(loadModelFile())
            var byteBuffer = convertBitmapToByteBuffer(resized)
            interpreter.run(byteBuffer, output)
            interpreter.close()
            chooseMax()
            outputConverted = convertOutputToString()
            arrayadapter = ArrayAdapter(
                this, android.R.layout.simple_list_item_1, outputConverted
            )
            listview.adapter = arrayadapter
            txtview.setText(shroomname)
            Log.d("pic", byteBuffer.toString())}
        })}


    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val MODEL_ASSETS_PATH = "model.tflite"
        val assetFileDescriptor = assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.getFileDescriptor())
        val fileChannel = fileInputStream.getChannel()
        val startoffset = assetFileDescriptor.getStartOffset()
        val declaredLength = assetFileDescriptor.getDeclaredLength()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * 50 * 50 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(50 * 50)
        bitmap.getPixels(
            intValues, 0, bitmap.width,
            0, 0, bitmap.width, bitmap.height
        )
        var pixel = 0
        for (i in 0..49) {
            for (j in 0..49) {
                val value = intValues[pixel++]
                byteBuffer.putFloat((value and 0xFF)/255.0.toFloat())
                Log.d("uwu1", ((value and 0xFF)/255.0.toFloat()).toString())
                byteBuffer.putFloat((value shr 8 and 0xFF)/255.0.toFloat())
                Log.d("uwu2", ((value shr 8 and 0xFF)/255.0.toFloat()).toString())
                byteBuffer.putFloat((value shr 16 and 0xFF)/255.0.toFloat())
                Log.d("uwu3", ((value shr 16 and 0xFF)/255.0.toFloat()).toString())
            }
        }
        return byteBuffer
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?)
    {
        if(data == null)
        {
            Log.d("Error", "No image attached")
        }
        else
        {
            if(requestCode==101&&resultCode== Activity.RESULT_OK)
            {
                bitmap = data?.extras?.get("data") as Bitmap
                if(bitmap != null)
                    imgview.setImageBitmap(bitmap)
            }
            else{
            isImageLoaded = true
            super.onActivityResult(requestCode, resultCode, data)
            var uri: Uri?=data?.data
            if(uri != null)
              bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            if(bitmap!=null)
              imgview.setImageURI(data?.data)}
        }
    }


}
