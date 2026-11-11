using UnityEngine;

public class SphereRotation : MonoBehaviour
{
    [Header("Rotation Settings")]
    [Tooltip("Speed of rotation in degrees per second")]
    public float rotationSpeed = 50f;

    void Update()
    {
        // Continuously rotate around the Y-axis (vertical)
        // Time.deltaTime ensures frame-rate independent smooth rotation
        transform.Rotate(0f, rotationSpeed * Time.deltaTime, 0f);
    }
}
