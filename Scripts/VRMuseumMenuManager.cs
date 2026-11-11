using UnityEngine;
using TMPro;

public class VRMuseumMenuManager : MonoBehaviour
{
    [Header("UI Panels")]
    public GameObject mainPanel;
    public GameObject exhibitInfoPanel;

    [Header("Feedback Display")]
    public TextMeshProUGUI statusFeedbackText;

    void Start()
    {
        if (mainPanel != null) mainPanel.SetActive(true);
        if (exhibitInfoPanel != null) exhibitInfoPanel.SetActive(false);
        UpdateStatus("🏛️ Museum Lobby: Aim ray or gaze to select an exhibit.");
    }

    public void OnStartTourClicked()
    {
        UpdateStatus("▶ Starting Guided Tour: Navigating to Exhibit #1...");
        Debug.Log("[VR UI] Tour Started.");
    }

    public void OnInformationClicked()
    {
        bool currentState = exhibitInfoPanel.activeSelf;
        exhibitInfoPanel.SetActive(!currentState);
        
        if (!currentState)
            UpdateStatus("ℹ️ Displaying Exhibit Specifications: Multi-Agent Digital Twin.");
        else
            UpdateStatus("ℹ️ Info Panel Closed.");
    }

    public void OnBackClicked()
    {
        if (exhibitInfoPanel != null) exhibitInfoPanel.SetActive(false);
        if (mainPanel != null) mainPanel.SetActive(true);
        UpdateStatus("◀ Returned to Main Museum Lobby.");
    }

    private void UpdateStatus(string message)
    {
        if (statusFeedbackText != null)
            statusFeedbackText.text = message;
    }
}
