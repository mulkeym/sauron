{{/*
Expand the name of the chart.
*/}}
{{- define "sauron.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sauron.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label
*/}}
{{- define "sauron.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sauron.labels" -}}
helm.sh/chart: {{ include "sauron.chart" . }}
{{ include "sauron.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sauron.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sauron.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "sauron.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sauron.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Full image reference (Harbor-friendly)
*/}}
{{- define "sauron.image" -}}
{{- $registry := .Values.image.registry | default "" | trimSuffix "/" -}}
{{- $repo := .Values.image.repository | required "image.repository is required" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion | default "latest" -}}
{{- $digest := .Values.image.digest | default "" -}}
{{- if $digest }}
{{- if $registry }}
{{- printf "%s/%s@%s" $registry $repo $digest }}
{{- else }}
{{- printf "%s@%s" $repo $digest }}
{{- end }}
{{- else if $registry }}
{{- printf "%s/%s:%s" $registry $repo $tag }}
{{- else }}
{{- printf "%s:%s" $repo $tag }}
{{- end }}
{{- end }}

{{/*
Secret name for app credentials
*/}}
{{- define "sauron.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "sauron.fullname" .) }}
{{- end }}
{{- end }}

{{/*
PVC name
*/}}
{{- define "sauron.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- printf "%s-data" (include "sauron.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Image pull secret names list (YAML)
*/}}
{{- define "sauron.imagePullSecretNames" -}}
{{- $names := list -}}
{{- range .Values.imagePullSecrets }}
{{- $names = append $names .name -}}
{{- end }}
{{- if and .Values.imagePullSecretsCreate.enabled .Values.imagePullSecretsCreate.name }}
{{- $names = append $names .Values.imagePullSecretsCreate.name -}}
{{- end }}
{{- $names | toJson }}
{{- end }}
