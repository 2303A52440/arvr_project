using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class VRGazeDwellButton : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
{
    [Header("Dwell Settings")]
    public float requiredDwellTime = 1.0f; // 1 second hover to click
    public Image dwellRadialFill;

    private float currentTimer = 0f;
    private bool isHovered = false;
    private Button targetButton;

    void Awake()
    {
        targetButton = GetComponent<Button>();
    }

    public void OnPointerEnter(PointerEventData eventData)
    {
        isHovered = true;
        currentTimer = 0f;
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        isHovered = false;
        currentTimer = 0f;
        if (dwellRadialFill != null) dwellRadialFill.fillAmount = 0f;
    }

    void Update()
    {
        if (!isHovered) return;

        currentTimer += Time.deltaTime;
        
        if (dwellRadialFill != null)
            dwellRadialFill.fillAmount = Mathf.Clamp01(currentTimer / requiredDwellTime);

        // When dwell reaches 1.0 second, trigger button click
        if (currentTimer >= requiredDwellTime)
        {
            if (targetButton != null)
                targetButton.onClick.Invoke();

            OnPointerExit(null); // Reset after activation
        }
    }
}
