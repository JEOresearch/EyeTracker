using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

//For use with Orlosky3DEyeTracker. Reads gaze origin and direction from file each frame
public class GazeFollower : MonoBehaviour
{
    public string filePath = "gaze_vector.txt";  // File in root project folder or specify absolute path

    void Update()
    {
        try
        {
            string[] values = File.ReadAllText(filePath).Split(',');
            if (values.Length != 6)
                return;

            // Parse gaze center and direction
            Vector3 gazeOrigin = new Vector3(
                float.Parse(values[0]),
                float.Parse(values[1]),
                float.Parse(values[2])
            );

            Vector3 gazeDirection = new Vector3(
                float.Parse(values[3]),
                float.Parse(values[4]),
                float.Parse(values[5])
            );
            
            // Update position to gaze origin
            transform.position = gazeOrigin;

            // Update rotation to face along gaze direction
            if (gazeDirection != Vector3.zero)
                transform.rotation = Quaternion.LookRotation(gazeDirection);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("Failed to read gaze vector file: " + e.Message);
        }
    }
}
