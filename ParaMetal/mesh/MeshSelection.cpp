#include "MeshSelection.hpp"

#include <algorithm>
#include <charconv>
#include <sstream>
#include <system_error>

static bool parseUint64(const std::string& text, uint64_t& outValue) {
    if (text.empty()) {
        return false;
    }
    uint64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end) {
        return false;
    }
    outValue = value;
    return true;
}

static bool splitRanges(
    const std::string& text,
    std::vector<uint64_t>& outElements,
    std::string& outError) {
    outElements.clear();
    if (text.empty()) {
        return true;
    }

    std::size_t tokenStart = 0;
    while (tokenStart <= text.size()) {
        const std::size_t comma = text.find(',', tokenStart);
        const std::string token = text.substr(
            tokenStart,
            comma == std::string::npos ? std::string::npos : comma - tokenStart);
        if (token.empty()) {
            outError = "Selection contains an empty range token.";
            return false;
        }

        const std::size_t dash = token.find('-');
        if (dash == std::string::npos) {
            uint64_t value = 0;
            if (!parseUint64(token, value)) {
                outError = "Selection contains an invalid element id.";
                return false;
            }
            outElements.push_back(value);
        } else {
            uint64_t first = 0;
            uint64_t last = 0;
            if (!parseUint64(token.substr(0, dash), first) ||
                !parseUint64(token.substr(dash + 1), last) ||
                last < first) {
                outError = "Selection contains an invalid element range.";
                return false;
            }
            if (last == UINT64_MAX || last - first > 100000000ull) {
                outError = "Selection range is unreasonably large.";
                return false;
            }
            for (uint64_t value = first; value <= last; ++value) {
                outElements.push_back(value);
            }
        }

        if (comma == std::string::npos) {
            break;
        }
        tokenStart = comma + 1;
    }
    return true;
}

std::string encodeMeshSelection(const MeshSelection& selection) {
    std::ostringstream text;
    text << "v1|type=" << static_cast<int>(selection.elementType)
         << "|count=" << selection.elements.size() << '|';

    bool first = true;
    for (std::size_t index = 0; index < selection.elements.size();) {
        std::size_t end = index + 1;
        while (end < selection.elements.size() &&
               selection.elements[end] == selection.elements[end - 1] + 1) {
            ++end;
        }
        if (!first) {
            text << ',';
        }
        first = false;
        if (end == index + 1) {
            text << selection.elements[index];
        } else {
            text << selection.elements[index] << '-' << selection.elements[end - 1];
        }
        index = end;
    }
    return text.str();
}

bool decodeMeshSelection(
    const std::string& text,
    MeshSelection& outSelection,
    std::string& outError) {
    outSelection = {};
    outError.clear();
    if (text.empty()) {
        return true;
    }

    std::size_t offset = 0;
    const auto nextField = [&]() -> std::string {
        const std::size_t delimiter = text.find('|', offset);
        const std::string field = text.substr(
            offset,
            delimiter == std::string::npos ? std::string::npos : delimiter - offset);
        offset = delimiter == std::string::npos ? text.size() + 1 : delimiter + 1;
        return field;
    };

    const std::string version = nextField();
    const std::string typeField = nextField();
    const std::string countField = nextField();
    if (version != "v1" || typeField.rfind("type=", 0) != 0 ||
        countField.rfind("count=", 0) != 0 || offset > text.size()) {
        outError = "Selection header is malformed.";
        return false;
    }

    uint64_t typeValue = 0;
    uint64_t expectedCount = 0;
    if (!parseUint64(typeField.substr(5), typeValue) ||
        !parseUint64(countField.substr(6), expectedCount) ||
        typeValue > static_cast<uint64_t>(MeshElementType::Face)) {
        outError = "Selection header contains an invalid type or count.";
        return false;
    }

    const std::string ranges = text.substr(std::min(offset, text.size()));
    if (!splitRanges(ranges, outSelection.elements, outError)) {
        return false;
    }
    if (outSelection.elements.size() != expectedCount) {
        outError = "Selection element count does not match its header.";
        return false;
    }

    for (std::size_t index = 1; index < outSelection.elements.size(); ++index) {
        if (outSelection.elements[index] <= outSelection.elements[index - 1]) {
            outError = "Selection elements must be sorted and unique.";
            return false;
        }
    }

    outSelection.elementType = static_cast<MeshElementType>(typeValue);
    return true;
}


