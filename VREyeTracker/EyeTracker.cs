using System;
using System.Globalization;
using System.IO;
using UnityEngine;

public class EyeTracker : MonoBehaviour
{
    private const string GazeFilePath = @"C:\users\jason\gaze_vector.txt";

    [Header("Visualization")]
    public float sphereDistance = 1.0f;
    public float sphereRadius = 0.05f;
    public float readInterval = 0.01f;

    [Header("Calibration")]
    public float verticalCalibrationAngleDegrees = 10.0f;

    [Header("Debug")]
    public bool verboseLogging = true;

    private GameObject gazeSphere;
    private GameObject calibrationSphere;

    private float nextReadTime = 0f;

    private Vector3 pythonOrigin;
    private Vector3 pythonDirection;

    private Vector3 latestRawUnityLocalDirection = Vector3.forward;
    private bool hasValidRawDirection = false;

    private bool hasCalibration = false;
    private Quaternion calibrationRotation = Quaternion.identity;

    private bool hasVerticalScale = false;
    private float verticalScaleFactor = 1.0f;

    private float upObservedPitchDegrees = 0f;
    private float downObservedPitchDegrees = 0f;

    private enum CalibrationStage
    {
        Idle,
        CenterTargetVisible,
        UpTargetVisible,
        DownTargetVisible
    }

    private CalibrationStage calibrationStage = CalibrationStage.Idle;

    private void Start()
    {
        CreateGazeSphere();
    }

    private void Update()
    {
        if (Time.time >= nextReadTime)
        {
            nextReadTime = Time.time + readInterval;
            ReadAndApplyGaze();
        }

        if (Input.GetKeyDown(KeyCode.B))
        {
            CreateFrozenSphereCopy();
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            HandleCalibrationKey();
        }
    }

    private void HandleCalibrationKey()
    {
        switch (calibrationStage)
        {
            case CalibrationStage.Idle:
                ShowCalibrationSphere(Vector3.forward, "CenterCalibrationSphere");
                calibrationStage = CalibrationStage.CenterTargetVisible;

                if (verboseLogging)
                    Debug.Log("Center target shown. Look at center sphere and press C again.");
                break;

            case CalibrationStage.CenterTargetVisible:
                CaptureCenterCalibration();
                ShowCalibrationSphere(DirectionFromPitchDegrees(verticalCalibrationAngleDegrees), "UpCalibrationSphere");
                calibrationStage = CalibrationStage.UpTargetVisible;

                if (verboseLogging)
                    Debug.Log("Center captured. Up target shown. Look at up sphere and press C again.");
                break;

            case CalibrationStage.UpTargetVisible:
                CaptureUpCalibration();
                ShowCalibrationSphere(DirectionFromPitchDegrees(-verticalCalibrationAngleDegrees), "DownCalibrationSphere");
                calibrationStage = CalibrationStage.DownTargetVisible;

                if (verboseLogging)
                    Debug.Log("Up captured. Down target shown. Look at down sphere and press C again.");
                break;

            case CalibrationStage.DownTargetVisible:
                CaptureDownCalibration();
                ComputeVerticalScaleFactor();
                ClearCalibrationSphere();
                calibrationStage = CalibrationStage.Idle;

                if (verboseLogging)
                    Debug.Log("Vertical calibration complete. Scale factor = " + verticalScaleFactor);
                break;
        }
    }

    private void CaptureCenterCalibration()
    {
        if (!hasValidRawDirection)
        {
            Debug.LogWarning("Center calibration failed: no valid raw gaze direction.");
            return;
        }

        Vector3 rawDir = latestRawUnityLocalDirection.normalized;

        calibrationRotation = Quaternion.FromToRotation(rawDir, Vector3.forward);
        hasCalibration = true;

        // Reset vertical scaling whenever a new center calibration is captured.
        hasVerticalScale = false;
        verticalScaleFactor = 1.0f;

        if (verboseLogging)
        {
            Debug.Log("Center calibration captured.");
            Debug.Log("Raw center direction: " + rawDir);
            Debug.Log("Calibration rotation: " + calibrationRotation.eulerAngles);
        }
    }

    private void CaptureUpCalibration()
    {
        if (!hasCalibration || !hasValidRawDirection)
        {
            Debug.LogWarning("Up calibration failed: missing center calibration or raw direction.");
            return;
        }

        Vector3 centeredDirection = (calibrationRotation * latestRawUnityLocalDirection).normalized;
        upObservedPitchDegrees = GetPitchDegrees(centeredDirection);

        if (verboseLogging)
            Debug.Log("Up observed pitch degrees: " + upObservedPitchDegrees);
    }

    private void CaptureDownCalibration()
    {
        if (!hasCalibration || !hasValidRawDirection)
        {
            Debug.LogWarning("Down calibration failed: missing center calibration or raw direction.");
            return;
        }

        Vector3 centeredDirection = (calibrationRotation * latestRawUnityLocalDirection).normalized;
        downObservedPitchDegrees = GetPitchDegrees(centeredDirection);

        if (verboseLogging)
            Debug.Log("Down observed pitch degrees: " + downObservedPitchDegrees);
    }

    private void ComputeVerticalScaleFactor()
    {
        float targetUp = verticalCalibrationAngleDegrees;
        float targetDown = -verticalCalibrationAngleDegrees;

        bool upValid = Mathf.Abs(upObservedPitchDegrees) > 0.001f;
        bool downValid = Mathf.Abs(downObservedPitchDegrees) > 0.001f;

        float sum = 0f;
        int count = 0;

        if (upValid)
        {
            float upScale = targetUp / upObservedPitchDegrees;
            sum += upScale;
            count++;

            if (verboseLogging)
                Debug.Log("Up scale estimate: " + upScale);
        }

        if (downValid)
        {
            float downScale = targetDown / downObservedPitchDegrees;
            sum += downScale;
            count++;

            if (verboseLogging)
                Debug.Log("Down scale estimate: " + downScale);
        }

        if (count > 0)
        {
            verticalScaleFactor = sum / count;
            hasVerticalScale = true;
        }
        else
        {
            verticalScaleFactor = 1.0f;
            hasVerticalScale = false;
            Debug.LogWarning("Vertical calibration failed: observed pitch values were too small.");
        }
    }

    private void ShowCalibrationSphere(Vector3 localDirection, string sphereName)
    {
        ClearCalibrationSphere();

        calibrationSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        calibrationSphere.name = sphereName;
        calibrationSphere.transform.SetParent(transform, false);

        float diameter = sphereRadius * 2.0f;
        calibrationSphere.transform.localScale = new Vector3(diameter, diameter, diameter);
        calibrationSphere.transform.localPosition = localDirection.normalized * sphereDistance;
        calibrationSphere.transform.localRotation = Quaternion.identity;

        Renderer renderer = calibrationSphere.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.color = Color.red;
        }
    }

    private void ClearCalibrationSphere()
    {
        if (calibrationSphere != null)
        {
            Destroy(calibrationSphere);
            calibrationSphere = null;
        }
    }

    private Vector3 DirectionFromPitchDegrees(float pitchDegrees)
    {
        float pitchRad = pitchDegrees * Mathf.Deg2Rad;

        return new Vector3(
            0f,
            Mathf.Sin(pitchRad),
            Mathf.Cos(pitchRad)
        ).normalized;
    }

    private float GetPitchDegrees(Vector3 direction)
    {
        direction.Normalize();

        float horizontalMagnitude = new Vector2(direction.x, direction.z).magnitude;
        float pitchRad = Mathf.Atan2(direction.y, horizontalMagnitude);

        return pitchRad * Mathf.Rad2Deg;
    }

    private Vector3 ApplyVerticalScale(Vector3 direction)
    {
        direction.Normalize();

        float yawRad = Mathf.Atan2(direction.x, direction.z);
        float horizontalMagnitude = new Vector2(direction.x, direction.z).magnitude;
        float pitchRad = Mathf.Atan2(direction.y, horizontalMagnitude);

        float scaledPitchRad = pitchRad * verticalScaleFactor;

        float newHorizontalMagnitude = Mathf.Cos(scaledPitchRad);

        Vector3 scaledDirection = new Vector3(
            Mathf.Sin(yawRad) * newHorizontalMagnitude,
            Mathf.Sin(scaledPitchRad),
            Mathf.Cos(yawRad) * newHorizontalMagnitude
        );

        return scaledDirection.normalized;
    }

    private void CreateFrozenSphereCopy()
    {
        if (gazeSphere == null)
            return;

        GameObject frozenSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        frozenSphere.name = "PythonGazeSphere_Frozen";
        frozenSphere.transform.position = gazeSphere.transform.position;
        frozenSphere.transform.rotation = gazeSphere.transform.rotation;
        frozenSphere.transform.localScale = gazeSphere.transform.lossyScale;
        frozenSphere.transform.SetParent(null, true);
    }

    private void CreateGazeSphere()
    {
        gazeSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        gazeSphere.name = "PythonGazeSphere";

        gazeSphere.transform.SetParent(transform, false);

        float diameter = sphereRadius * 2.0f;
        gazeSphere.transform.localScale = new Vector3(diameter, diameter, diameter);
        gazeSphere.transform.localPosition = Vector3.forward * sphereDistance;
        gazeSphere.transform.localRotation = Quaternion.identity;
    }

    private void ReadAndApplyGaze()
    {
        string rawText = ReadTextFileSafe(GazeFilePath);
        if (string.IsNullOrWhiteSpace(rawText))
            return;

        if (!TryParseSixFloats(rawText, out float[] values))
            return;

        pythonOrigin = new Vector3(values[0], values[1], values[2]);
        pythonDirection = new Vector3(values[3], values[4], values[5]);

        Vector3 unityLocalDirection = PythonToUnityDirection(pythonDirection);

        if (unityLocalDirection.sqrMagnitude < 0.000001f)
        {
            hasValidRawDirection = false;
            return;
        }

        unityLocalDirection.Normalize();

        latestRawUnityLocalDirection = unityLocalDirection;
        hasValidRawDirection = true;

        Vector3 correctedDirection = hasCalibration
            ? (calibrationRotation * unityLocalDirection).normalized
            : unityLocalDirection;

        if (hasVerticalScale)
        {
            correctedDirection = ApplyVerticalScale(correctedDirection);
        }

        gazeSphere.transform.localPosition = correctedDirection * sphereDistance;
    }

    private Vector3 PythonToUnityDirection(Vector3 pythonVec)
    {
        return new Vector3(
            pythonVec.x,
            pythonVec.y,
            pythonVec.z
        );
    }

    private string ReadTextFileSafe(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                Debug.LogWarning("Gaze file not found: " + path);
                return null;
            }

            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (StreamReader reader = new StreamReader(fs))
            {
                return reader.ReadToEnd();
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("Failed to read gaze file: " + e.Message);
            return null;
        }
    }

    private bool TryParseSixFloats(string text, out float[] values)
    {
        values = null;

        string[] tokens = text.Split(
            new char[] { ',', ' ', '\t', '\r', '\n', ';' },
            StringSplitOptions.RemoveEmptyEntries
        );

        if (tokens.Length < 6)
            return false;

        values = new float[6];

        for (int i = 0; i < 6; i++)
        {
            if (!float.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]))
            {
                values = null;
                return false;
            }
        }

        return true;
    }
}